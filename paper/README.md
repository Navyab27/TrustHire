# TrustHire conference paper (IEEE format)

`main.tex` is an IEEEtran conference paper; `main.pdf` is the compiled version.

## Build

- **Overleaf:** upload this whole `paper/` folder (main.tex, references.bib, numbers.tex, figures/) and compile
  with pdfLaTeX (default) or XeLaTeX.
- **Locally:** `tectonic -X compile main.tex` (or `latexmk -pdf main.tex`).

## Regenerating numbers and figures

Every number quoted in the paper comes from `numbers.tex`, which is generated from `results/` together with all
plots:

```bash
python paper/make_figures.py          # figures/*.pdf|png, figures/stats.json, numbers.tex
python paper/make_gradcam_figure.py   # figures/gradcam.png
```

## Before submission

- Replace the placeholders in the author blocks (`Name of the Institute`, `City`, e-mail addresses).
- Keep or adapt the AI-assistance disclosure in the Acknowledgment; IEEE requires it when AI tools were used.
