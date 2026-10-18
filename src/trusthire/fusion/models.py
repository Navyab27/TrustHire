"""Fusion models: (a) weighted late fusion, (b) gradient-boosted stacker, (c) cross-modal attention.

All share `fit(df)`, `predict_proba(df) -> P(fraud)` and `explain(df)` (per-module contributions), and are
wrapped by a post-hoc calibrator (temperature scaling or isotonic regression).
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from torch import nn

from trusthire.fusion.features import MODS, add_temporal_context, conf_cols, raw_cols, score_cols

# Prior trust in each module (used before any labelled data exists; refit by WeightedLateFusion.fit)
PRIOR_WEIGHTS = {
    "video_deepfake": 1.0,
    "voice_spoof": 1.0,
    "av_sync": 0.8,
    "face_identity": 1.3,
    "voice_identity": 1.0,
    "behavior": 0.6,
    "liveness": 0.3,
    "objects": 0.9,
    "scripted_answer": 0.6,
    "resume_consistency": 0.5,
}
EPS = 1e-4


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.asarray(np.log(p / (1 - p)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return np.asarray(1 / (1 + np.exp(-z)))


class Fuser(Protocol):
    name: str

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> Fuser: ...
    def predict_logit(self, df: pd.DataFrame) -> np.ndarray: ...
    def contributions(self, df: pd.DataFrame) -> pd.DataFrame: ...


# =============================================================================== (a) weighted late fusion
class WeightedLateFusion:
    """Confidence-weighted log-odds pooling with an explicit "red flag" path:

        logit(risk) = b + Σ_m w_m c_m ℓ_m / Σ_m w_m c_m   +   Σ_m v_m c_m · max(0, ℓ_m)

    where ℓ_m = logit(s_m). The first term is a weighted *mean* (correlated modules agreeing that the
    candidate looks genuine do not compound into absurd certainty, unlike a naive-Bayes sum); the second
    lets one confident suspicious module (e.g. a face mismatch) dominate. Missing modules contribute
    nothing. w, v ≥ 0 and b are fit by class-balanced, L2-to-prior regularised logistic loss.
    """

    name = "weighted"

    def __init__(self, mods: tuple[str, ...] = MODS, l2: float = 0.05):
        self.mods = mods
        self.l2 = l2
        self.w0 = np.array([PRIOR_WEIGHTS.get(m, 0.5) for m in mods])
        self.v0 = 0.45 * self.w0
        self.w, self.v, self.b = self.w0.copy(), self.v0.copy(), 0.0

    def _parts(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        s = df[score_cols(self.mods)].to_numpy(dtype=float)
        c = df[conf_cols(self.mods)].to_numpy(dtype=float)
        miss = np.isnan(s)
        c = np.where(miss, 0.0, c)
        ell = np.where(miss, 0.0, _logit(np.nan_to_num(s, nan=0.5)))
        return c, ell

    def _logit_and_terms(
        self, c: np.ndarray, ell: np.ndarray, w: np.ndarray, v: np.ndarray, b: float
    ) -> tuple[np.ndarray, np.ndarray]:
        wc = c * w
        denom = wc.sum(1, keepdims=True)
        mean_terms = np.divide(wc * ell, denom, out=np.zeros_like(ell), where=denom > 1e-9)
        flag_terms = c * v * np.maximum(ell, 0.0)
        terms = mean_terms + flag_terms
        return terms.sum(1) + b, terms

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> WeightedLateFusion:
        c, ell = self._parts(df)
        yy = y.astype(float)
        pos_w = (1 - yy.mean()) / max(yy.mean(), 1e-3)  # balance classes
        sw = np.where(yy == 1, pos_w, 1.0)
        m = len(self.mods)

        def loss(theta: np.ndarray) -> float:
            w, v, b = theta[:m], theta[m : 2 * m], theta[-1]
            z, _ = self._logit_and_terms(c, ell, w, v, b)
            ll = sw * (np.logaddexp(0, z) - yy * z)
            reg = np.sum((w - self.w0) ** 2) + np.sum((v - self.v0) ** 2)
            return float(ll.sum() / sw.sum() + self.l2 * reg)

        x0 = np.concatenate([self.w, self.v, [self.b]])
        bounds = [(1e-3, 5.0)] * m + [(0.0, 5.0)] * m + [(-5.0, 5.0)]
        res = minimize(loss, x0, method="L-BFGS-B", bounds=bounds)
        self.w, self.v, self.b = res.x[:m], res.x[m : 2 * m], float(res.x[-1])
        return self

    def predict_logit(self, df: pd.DataFrame) -> np.ndarray:
        c, ell = self._parts(df)
        return self._logit_and_terms(c, ell, self.w, self.v, self.b)[0]

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        c, ell = self._parts(df)
        _, terms = self._logit_and_terms(c, ell, self.w, self.v, self.b)
        return pd.DataFrame(terms, columns=list(self.mods), index=df.index)


# =============================================================================== (b) GBM stacker
class GBMStacker:
    """LightGBM (default) or XGBoost over module scores, confidences, raw measurements and causal temporal
    context. NaN = missing modality (handled natively by both libraries). SHAP gives contributions."""

    name = "gbm"

    def __init__(
        self, mods: tuple[str, ...] = MODS, backend: str = "lightgbm", temporal: bool = True, seed: int = 0
    ):
        self.mods = mods
        self.backend = backend
        self.temporal = temporal
        self.seed = seed
        self.model: Any = None
        self.cols: list[str] = []

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        d = add_temporal_context(df, self.mods) if self.temporal and "session_id" in df else df
        cols = score_cols(self.mods) + conf_cols(self.mods) + raw_cols(self.mods)
        if self.temporal and "session_id" in df:
            cols += [c + s for c in score_cols(self.mods) for s in ("_rmean", "_rmax")]
        self.cols = self.cols or cols
        return d.reindex(columns=self.cols).astype(float)

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> GBMStacker:
        self.cols = []
        X = self._X(df)
        spw = float((y == 0).sum() / max((y == 1).sum(), 1))
        if self.backend == "xgboost":
            import xgboost as xgb

            self.model = xgb.XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw,
                random_state=self.seed,
                eval_metric="logloss",
            )
        else:
            import lightgbm as lgb

            self.model = lgb.LGBMClassifier(
                n_estimators=300,
                num_leaves=15,
                learning_rate=0.05,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.8,
                min_child_samples=10,
                n_jobs=1 if sys.platform == "darwin" else 4,
                scale_pos_weight=spw,
                random_state=self.seed,
                verbose=-1,
            )
        self.model.fit(X, y)
        return self

    def predict_logit(self, df: pd.DataFrame) -> np.ndarray:
        p = np.clip(self.model.predict_proba(self._X(df))[:, 1], 1e-12, 1 - 1e-12)
        return np.asarray(np.log(p) - np.log1p(-p))  # no coarse clipping: ties would destroy the ranking

    def shap_values(self, df: pd.DataFrame) -> pd.DataFrame:
        import shap

        X = self._X(df)
        sv = shap.TreeExplainer(self.model).shap_values(X)
        sv = sv[1] if isinstance(sv, list) else sv
        return pd.DataFrame(sv, columns=X.columns, index=df.index)

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        sv = self.shap_values(df)
        out = pd.DataFrame(index=df.index)
        for m in self.mods:
            out[m] = sv[[c for c in sv.columns if c.startswith(m) or self._raw_owner(c) == m]].sum(1)
        return out

    @staticmethod
    def _raw_owner(col: str) -> str | None:
        from trusthire.fusion.features import RAW_FEATURES

        return RAW_FEATURES[col][0] if col in RAW_FEATURES else None


# =============================================================================== (c) cross-modal attention
class CrossModalAttentionNet(nn.Module):
    """Each module becomes a token: Linear([score, conf]) + learned modality embedding. Missing modules are
    masked out of attention. Tokens from the previous T-1 windows are included (with time embeddings), so
    the model attends across modalities *and* time. A CLS token produces the window logit."""

    def __init__(
        self,
        n_mods: int,
        d: int = 32,
        heads: int = 4,
        layers: int = 2,
        context: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_mods = n_mods
        self.context = context
        self.inp = nn.Sequential(nn.Linear(3, d), nn.GELU(), nn.Linear(d, d))
        self.mod_emb = nn.Parameter(torch.randn(n_mods, d) * 0.02)
        self.time_emb = nn.Parameter(torch.randn(context, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(
        self, s: torch.Tensor, c: torch.Tensor, present: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """s, c, present: (B, T, M). Returns logits (B,) and CLS attention over tokens (B, T*M)."""
        b, t, m = s.shape
        x = torch.stack([torch.logit(s.clamp(EPS, 1 - EPS)) * present, c * present, present], -1)
        tok = self.inp(x) + self.mod_emb[None, None] + self.time_emb[None, -t:, None]
        tok = tok.reshape(b, t * m, -1)
        pad = ~present.reshape(b, t * m).bool()
        h = torch.cat([self.cls.expand(b, -1, -1), tok], 1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=s.device), pad], 1)
        h = self.enc(h, src_key_padding_mask=pad)
        # attention proxy for explanation: similarity of CLS to each token (masked softmax)
        att = torch.einsum("bd,bnd->bn", h[:, 0], h[:, 1:]) / h.shape[-1] ** 0.5
        att = att.masked_fill(pad[:, 1:], -1e4).softmax(-1)
        return self.out(h[:, 0]).squeeze(-1), att


class AttentionFusion:
    name = "attention"

    def __init__(
        self,
        mods: tuple[str, ...] = MODS,
        context: int = 4,
        epochs: int = 30,
        lr: float = 5e-3,
        modality_dropout: float = 0.3,
        seed: int = 0,
    ):
        self.mods = mods
        self.context = context
        self.epochs = epochs
        self.lr = lr
        self.p_drop = modality_dropout
        self.seed = seed
        self.net = CrossModalAttentionNet(len(mods), context=context)

    def _tensors(self, df: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = df[score_cols(self.mods)].to_numpy(float)
        c = df[conf_cols(self.mods)].to_numpy(float)
        present = (~np.isnan(s)) & (c > 0)
        s = np.nan_to_num(s, nan=0.5)
        sessions = df["session_id"].to_numpy() if "session_id" in df else np.zeros(len(df))
        n, m, k = len(df), len(self.mods), self.context
        S, C, P = np.full((n, k, m), 0.5), np.zeros((n, k, m)), np.zeros((n, k, m), bool)
        for i in range(n):  # causal context window within the same session
            for j in range(k):
                src = i - (k - 1 - j)
                if src >= 0 and sessions[src] == sessions[i]:
                    S[i, j], C[i, j], P[i, j] = s[src], c[src], present[src]
        return (
            torch.tensor(S, dtype=torch.float32),
            torch.tensor(C, dtype=torch.float32),
            torch.tensor(P, dtype=torch.float32),
        )

    def fit(self, df: pd.DataFrame, y: np.ndarray) -> AttentionFusion:
        torch.manual_seed(self.seed)
        S, C, P = self._tensors(df)
        Y = torch.tensor(y, dtype=torch.float32)
        pos_w = torch.tensor((1 - y.mean()) / max(y.mean(), 1e-3), dtype=torch.float32)
        opt = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, self.epochs)
        g = torch.Generator().manual_seed(self.seed)
        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(len(Y), generator=g)
            for i in range(0, len(Y), 256):
                idx = perm[i : i + 256]
                p = P[idx].clone()
                # modality dropout: teaches robustness to missing modules (never drop every present token)
                drop = (torch.rand(p.shape[0], 1, p.shape[2], generator=g) < self.p_drop).float()
                keep = p * (1 - drop)
                empty = keep.sum((1, 2)) == 0
                keep[empty] = p[empty]
                logit, _ = self.net(S[idx], C[idx], keep)
                loss = nn.functional.binary_cross_entropy_with_logits(logit, Y[idx], pos_weight=pos_w)
                opt.zero_grad()
                loss.backward()
                opt.step()
            sched.step()
        self.net.eval()
        return self

    def predict_logit(self, df: pd.DataFrame) -> np.ndarray:
        S, C, P = self._tensors(df)
        with torch.inference_mode():
            logit, _ = self.net(S, C, P)
        return logit.numpy()

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attention mass on the current window's tokens, signed by each module's log-odds."""
        S, C, P = self._tensors(df)
        with torch.inference_mode():
            _, att = self.net(S, C, P)
        m = len(self.mods)
        cur = att[:, -m:].numpy()
        sign = _logit(S[:, -1].numpy()) * P[:, -1].numpy()
        return pd.DataFrame(cur * sign, columns=list(self.mods), index=df.index)


# =============================================================================== calibration
class Calibrator:
    """'temperature' (Guo et al. 2017) or 'isotonic' post-hoc calibration of fused logits."""

    def __init__(self, method: str = "temperature"):
        self.method = method
        self.T = 1.0
        self.iso: Any = None

    def fit(self, logits: np.ndarray, y: np.ndarray) -> Calibrator:
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression

            self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(logits, y)
        else:

            def nll(t: np.ndarray) -> float:
                z = logits / t[0]
                return float(np.mean(np.logaddexp(0, z) - y * z))

            self.T = float(minimize(nll, [1.0], bounds=[(0.05, 20.0)], method="L-BFGS-B").x[0])
        return self

    def __call__(self, logits: np.ndarray) -> np.ndarray:
        if self.method == "isotonic" and self.iso is not None:
            return np.asarray(self.iso.predict(logits))
        return _sigmoid(logits / self.T)


class CalibratedFuser:
    def __init__(self, fuser: Any, calibrator: Calibrator | None = None):
        self.fuser = fuser
        self.cal = calibrator or Calibrator("temperature")
        self.name = fuser.name

    def fit(
        self,
        df: pd.DataFrame,
        y: np.ndarray,
        df_cal: pd.DataFrame | None = None,
        y_cal: np.ndarray | None = None,
    ) -> CalibratedFuser:
        self.fuser.fit(df, y)
        if df_cal is not None and y_cal is not None and len(np.unique(y_cal)) == 2:
            self.cal.fit(self.fuser.predict_logit(df_cal), y_cal)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.cal(self.fuser.predict_logit(df))

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fuser.contributions(df)  # type: ignore[no-any-return]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str | Path) -> CalibratedFuser:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, CalibratedFuser)
        return obj


def build_fuser(method: str, mods: tuple[str, ...] = MODS, seed: int = 0) -> Any:
    if method == "weighted":
        return WeightedLateFusion(mods)
    if method in ("gbm", "lightgbm"):
        return GBMStacker(mods, "lightgbm", seed=seed)
    if method == "xgboost":
        return GBMStacker(mods, "xgboost", seed=seed)
    if method == "attention":
        return AttentionFusion(mods, seed=seed)
    raise ValueError(method)
