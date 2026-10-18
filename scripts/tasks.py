"""Cross-platform task runner (stdlib only). Used by the Makefile and by make.cmd on Windows.

python scripts/tasks.py demo        # one command: env + models + frontend build + server + browser
python scripts/tasks.py <task>      # see TASKS below
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home() / ".trusthire"
VENV = Path(os.environ.get("UV_PROJECT_ENVIRONMENT", HOME / "venv"))  # outside the repo / OneDrive
PY = VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
ENV = {
    **os.environ,
    "UV_PROJECT_ENVIRONMENT": str(VENV),
    "UV_LINK_MODE": "copy",
    "UV_SYSTEM_CERTS": "1",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
    "PYTHONIOENCODING": "utf-8",
    "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS", "1" if sys.platform == "darwin" else "4"),
}


def sh(*cmd: str | Path, cwd: Path = ROOT, check: bool = True) -> int:
    print("$", " ".join(map(str, cmd)), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=cwd, env=ENV, check=check).returncode


def uv(*args: str) -> int:
    exe = shutil.which("uv")
    return sh(*(exe,) if exe else (sys.executable, "-m", "uv"), *args)


def npm(*args: str) -> int:
    exe = shutil.which("npm") or shutil.which("npm.cmd")
    if not exe:
        raise SystemExit("npm not found — install Node.js >= 20")
    return sh(exe, *args, cwd=ROOT / "frontend")


def tm(*args: str) -> int:
    return sh(PY, "-m", *args)


# ------------------------------------------------------------------ tasks
def setup() -> None:
    if not (
        shutil.which("uv")
        or subprocess.run([sys.executable, "-m", "uv", "--version"], capture_output=True).returncode == 0
    ):
        sh(sys.executable, "-m", "pip", "install", "uv")
    uv("python", "install", "3.12")
    uv("sync", "--python", "3.12", "--extra", "track")
    npm("ci", "--no-audit", "--no-fund")


def models() -> None:
    tm("trusthire.cli", "download-models", "--profile", os.environ.get("TRUSTHIRE_PROFILE", "cpu"))


def frontend() -> None:
    if not (ROOT / "frontend" / "node_modules").exists():
        npm("ci", "--no-audit", "--no-fund")
    npm("run", "build")


def serve() -> None:
    tm("trusthire.cli", "serve", "--host", "127.0.0.1", "--port", "8000")


def demo() -> None:
    if not PY.exists():
        setup()
    models()
    if not (ROOT / "frontend" / "dist" / "index.html").exists():
        frontend()
    proc = subprocess.Popen([str(PY), "-m", "trusthire.cli", "serve", "--port", "8000"], cwd=ROOT, env=ENV)
    for _ in range(120):
        try:
            urllib.request.urlopen("http://127.0.0.1:8000/api/health", timeout=2)
            break
        except Exception:
            time.sleep(1)
    print("\n  TrustHire is running at http://127.0.0.1:8000  (Ctrl+C to stop)\n")
    webbrowser.open("http://127.0.0.1:8000")
    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


def dev() -> None:
    api = subprocess.Popen([str(PY), "-m", "trusthire.cli", "serve", "--reload"], cwd=ROOT, env=ENV)
    try:
        npm("run", "dev")
    finally:
        api.terminate()


def test() -> None:
    tm("pytest", "-q")


def lint() -> None:
    tm("ruff", "check", "src", "tests")
    tm("ruff", "format", "--check", "src", "tests")
    npm("run", "typecheck")


def typecheck() -> None:
    tm("mypy")


def bench() -> None:
    tm(
        "trusthire.audio.eval",
        "--datasets",
        "itw",
        "asv19la",
        "--n",
        "400",
        "--backends",
        "aasist",
        "aasist_l",
        "spectra0",
        "spectra0_int8",
    )
    tm(
        "trusthire.audio.eval",
        "--measure-latency",
        "--backends",
        "aasist",
        "aasist_l",
        "spectra0",
        "spectra0_int8",
    )
    tm("trusthire.identity.eval", "face")
    tm("trusthire.avsync.eval")
    tm("trusthire.evaluation.latency")


def eval_custom() -> None:
    tm("trusthire.evaluation.harness", "extract")
    tm("trusthire.evaluation.harness", "evaluate")
    tm("trusthire.evaluation.report")


def eval_synthetic() -> None:
    tm("trusthire.evaluation.harness", "evaluate", "--synthetic")


def report() -> None:
    tm("trusthire.evaluation.report")


def docker_up() -> None:
    sh("docker", "compose", "up", "--build")


TASKS = {
    "setup": setup,
    "models": models,
    "frontend": frontend,
    "serve": serve,
    "demo": demo,
    "dev": dev,
    "test": test,
    "lint": lint,
    "typecheck": typecheck,
    "bench": bench,
    "eval-custom": eval_custom,
    "eval-synthetic": eval_synthetic,
    "report": report,
    "docker-up": docker_up,
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in TASKS:
        print("usage: make <task>   tasks:", ", ".join(TASKS))
        raise SystemExit(2)
    TASKS[sys.argv[1]]()
