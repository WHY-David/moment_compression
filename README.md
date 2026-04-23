# moment_compression

Code accompanying [_A Universal Compression Theory for the Lottery Ticket Hypothesis and Neural Scaling Laws_](ICLR2026_A_Universal_Compression_Theory/ICLRver.tex) (Wang, Luo, Poggio, Chuang, Ziyin; ICLR 2026). The paper shows that any permutation-invariant function of `d` objects can be losslessly compressed (in the asymptotic sense) to `O(polylog d)` weighted objects by matching the first `k` tensor moments, which in turn gives the dynamical lottery ticket hypothesis and stretched-exponential improvements to neural scaling laws.

Companion docs: [PAPER_SUMMARY.md](PAPER_SUMMARY.md) (figure → script map), [CLEANUP_PLAN.md](CLEANUP_PLAN.md) (release-quality checklist).

---

## Install

Conda (recommended — matches the original `torch-env`, and the only way to get `faiss-cpu` for Python ≥ 3.13):

```bash
conda env create -f environment.yml      # creates `torch-env`
conda activate torch-env
```

Pip-only alternative (Python 3.10–3.12 only — the PyPI `faiss-cpu` wheel is not published for 3.13+):

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## How to use `compressor.py`

[compressor.py](compressor.py) is a single-file library at the repo root. Every experiment in this repo — and every downstream user — imports its two exports:

```python
from compressor import Compressor, multi_exponents, all_moments
```

### Minimum working example

Run from the repo root so `compressor.py` is on `sys.path`:

```python
import numpy as np
from compressor import Compressor, multi_exponents, all_moments

rng = np.random.default_rng(0)
data = rng.standard_normal((2000, 3))             # d=2000 points in R^m, here m=3

cp = Compressor(data, random_state=0)             # accepts (d, m) arrays; can also take weights=c
c, w = cp.compress(k=2, dstop=None)               # match every tensor moment up to order k=2

# c: length-d' weights (most entries zero); w: (d', m) surviving vectors.
exps = multi_exponents(3, 2)
orig = sum(all_moments(x, exps) for x in data)
comp = sum(cj * all_moments(wj, exps) for cj, wj in zip(c, w))
print("kept", c.size, "of", data.shape[0], "points; max moment error:", float(np.max(np.abs(orig - comp))))
```

Expected: `kept 10 of 2000` — that's `C(m+k, k) = C(5,2) = 10`, the Tchakaloff bound — and a moment error near `1e-10`.

### What the knobs do

- `Compressor(data, weights=None, random_state=0, tol=…)` — `data` is `(d, m)`; `weights` is a length-`d` non-negative vector (defaults to uniform).
- `cp.compress(k, dstop=…)`:
  - `k` — the highest tensor-moment order to preserve. Error bound from Thm. `compression_errorbound` in the paper: `|φ(c') − φ(c)| = O(d · d'^{1 − (k+1)/m})`.
  - `dstop` — target support size. If `None`, the Carathéodory-peeling step runs to `C(m+k, k)` (the theoretical minimum). Set it larger to stop earlier and trade accuracy for speed.
- Internally `compress` first runs sample-weighted `MiniBatchKMeans` while `|supp c| ≫ dstop`, then hands off to a FAISS nearest-neighbour greedy phase. IVF indexing kicks in above `d ≈ 10⁷`; below that, the code falls back to `IndexFlatL2` transparently.
- Extending: the two workhorse methods are `_find_best_subset` (the greedy cluster picker) and `_reduce_compute` (Carathéodory peeling via SVD null-vector). Replacing either is the main way to customise the algorithm without breaking the outer interface.

## Reproducing the figures

Each paper figure has its own top-level folder. Inside, `run script → CSV → plot notebook → PDF`:

| Figure (paper section)                                       | Folder                                           | Run script(s)                                                                                            | Plot notebook                                         |
| ------------------------------------------------------------ | ------------------------------------------------ | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Fig. 2 — Generic error scaling (Sec. 3.2)                    | [fig2_error_scaling/](fig2_error_scaling/)       | [error_scaling.py](fig2_error_scaling/error_scaling.py), [job_array.py](fig2_error_scaling/job_array.py) | [plot.ipynb](fig2_error_scaling/plot.ipynb)           |
| Fig. 3 — Compressing the training dataset (Sec. 3.2)         | [fig3_compress_trainds/](fig3_compress_trainds/) | [trainds.py](fig3_compress_trainds/trainds.py)                                                           | [plot.ipynb](fig3_compress_trainds/plot.ipynb)        |
| Fig. 4 — Dynamical lottery ticket hypothesis (Sec. 4)        | [fig4_lth/](fig4_lth/)                           | [compress_dynamics.py](fig4_lth/compress_dynamics.py)                                                    | [plot.ipynb](fig4_lth/plot.ipynb)                     |
| Fig. 5 — Improved neural scaling laws (Sec. 5)               | [fig5_scaling_law/](fig5_scaling_law/)           | [trainds.py](fig5_scaling_law/trainds.py), [width.py](fig5_scaling_law/width.py)                         | [plot.ipynb](fig5_scaling_law/plot.ipynb)             |
| Fig. 6 — Hybrid-algorithm runtime benchmark (Appendix)       | [fig6_runtime/](fig6_runtime/)                   | [runtime.py](fig6_runtime/runtime.py)                                                                    | [runtime_plot.ipynb](fig6_runtime/runtime_plot.ipynb) |
| Fig. 7 — Multi-head attention LTH (Appendix `app:attention`) | [fig7_attention/](fig7_attention/)               | [ICL_piecewise.py](fig7_attention/ICL_piecewise.py)                                                      | [plot.ipynb](fig7_attention/plot.ipynb)               |
| Fig. 8 — Polylog-compression error scaling (Appendix)        | [fig8_polylog/](fig8_polylog/)                   | [polylog_error.py](fig8_polylog/polylog_error.py)                                                        | [plot.ipynb](fig8_polylog/plot.ipynb)                 |

Run a figure's script from inside its folder; it writes CSVs into a local subfolder. Open the folder's `plot.ipynb` to regenerate the PDF.

The quickest paths:

- **Fig. 6 runtime** (minutes on a laptop): `cd fig6_runtime && python runtime.py` → `time_list.csv`, then open `runtime_plot.ipynb`.
- **Fig. 2 error scaling** (dial down `d_list`, `mlist`, `k_list`, `trials_per_d` at the bottom of the script for a fast preview): `cd fig2_error_scaling && python error_scaling.py`.
- **Figs. 5** is a Slurm-array sweep over `(d, seed)`; see the `sid = int(os.environ["SLURM_ARRAY_TASK_ID"])` pattern in [trainds.py](fig5_scaling_law/trainds.py) and [width.py](fig5_scaling_law/width.py). For a single-seed preview, replace the `sid` block with hard-coded `seed, d`.

## `demo/`

[demo/](demo/) is a self-contained tutorial folder (renamed from the anonymous-submission supplementary). It shows basic use cases of `Compressor` at small scale, so every script here finishes on a laptop CPU:

- [demo/demo.py](demo/demo.py) — visualises 2-D and 3-D random-point clouds and their weighted-object compression with `dstop = C(m+k, k)`. Running `python demo/demo.py` pops a side-by-side scatter of 1000 3-D Gaussian points and their 40-atom moment-matching compression (marker size ∝ weight).
- [demo/trainds.py](demo/trainds.py) — small-scale version of Fig. 3 (compressing the training dataset). Hyperparameters are at lines 103–118.
- [demo/compress_dynamics.py](demo/compress_dynamics.py) — small-scale version of Fig. 4 (dynamical lottery ticket hypothesis). Hyperparameters are at lines 137–183.

## Dependencies at a glance

| Library                 | Why it's needed                                                                                                                                                                                                   |
| ----------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `numpy`                 | Everything.                                                                                                                                                                                                       |
| `scipy`                 | `scipy.special.jv` (cylindrical-harmonic target in Figs. 4, 5); `scipy.optimize.curve_fit` in Fig. 5's plotting notebook.                                                                                         |
| `scikit-learn`          | `MiniBatchKMeans` clustering inside `Compressor.compress` when `alive ≫ dstop`.                                                                                                                                   |
| `joblib`                | Parallel `_reduce_compute` over clusters.                                                                                                                                                                         |
| `faiss-cpu`             | `IndexFlatL2` / `IndexIDMap2` for the greedy nearest-neighbour subset search.                                                                                                                                     |
| `torch` (≥ 2.3)         | All training experiments. ≥ 2.3 is required because [fig5_scaling_law/width.py](fig5_scaling_law/width.py) imports `torch.amp.{autocast, GradScaler}` and `common.py` touches `torch.backends.mps.deterministic`. |
| `matplotlib`            | Every figure. `common.py::make_canvas` sets `pdf.use14corefonts=True`, so no custom fonts are installed.                                                                                                          |
| `pandas`                | CSV I/O in the plotting notebooks.                                                                                                                                                                                |
| `jupyter` / `ipykernel` | Running the `plot.ipynb` notebooks.                                                                                                                                                                               |

No `pandas` imports exist in [compressor.py](compressor.py) itself — the core library runs on `numpy + scipy + scikit-learn + joblib + faiss-cpu`.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'compressor'`** — run from the repo root, or run the experiment scripts from inside their `figN_*` folder (each figure script has a `sys.path.insert(…, '..')` shim that reaches the root `compressor.py`).
- **`faiss` install fails on Python 3.13** — downgrade to 3.12 (conda wheels). 3.13 wheels are not yet published as of this writing.
- **`torch.backends.mps.deterministic` AttributeError** — PyTorch older than 2.1; upgrade to ≥ 2.3 to match the pinned environment.
- **Out-of-memory during `Compressor.compress` at high `k`** — the moment matrix is `C(m+k, k) × d`. Drop `k`, or set `dstop` larger so more work stays in the cheap k-means phase.
- **IVF fallback message `[fallback] IVF search failed …`** — harmless; the code switches itself to `IndexFlatL2` and retries. IVF is only beneficial at `d ≳ 10⁷`.

## Citations

- OpenReview forum: <https://openreview.net/forum?id=vxkzW4ljeX&nesting=2&sort=date-desc>
- arXiv preprint: <https://arxiv.org/abs/2510.00504>

```bibtex
@misc{wang2025universalcompression,
  title         = {A Universal Compression Theory for the Lottery Ticket Hypothesis and Neural Scaling Laws},
  author        = {Hong-Yi Wang and Di Luo and Tomaso Poggio and Isaac L. Chuang and Liu Ziyin},
  year          = {2025},
  eprint        = {2510.00504},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2510.00504}
}
```

---

This repository was initially written by Hong-Yi Wang (hywang@princeton.edu), and later reorganized using Claude Code.
