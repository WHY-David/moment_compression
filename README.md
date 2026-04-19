# moment_compression

Code for the paper [_A Universal Compression Theory for the Lottery Ticket Hypothesis and Neural Scaling Laws_](ICLR2026_A_Universal_Compression_Theory/ICLRver.tex) (Wang, Luo, Poggio, Chuang, Ziyin; ICLR 2026).

The core export is `Compressor` in [compressor.py](compressor.py), which compresses a set of `d` vectors in `R^m` to at most `C(m+k, k)` weighted vectors while preserving all tensor moments up to order `k`. Everything else in the repo is either an experiment built on top of this primitive or a plotting notebook.

- Paper + figure-to-code map: [PAPER_SUMMARY.md](PAPER_SUMMARY.md)
- Plan for turning this research tree into a release-quality repo: [CLEANUP_PLAN.md](CLEANUP_PLAN.md)

## Using the compressor

```python
from compressor import Compressor

cp = Compressor(data, random_state=0)            # data: (d, m) float array
weights, compressed_vecs = cp.compress(k, dstop=...)
```

`Compressor` has a fixed RNG seed by default. IVF indexing becomes faster than flat at roughly `d ≳ 10^7`; below that, the code transparently falls back to `IndexFlatL2`.

## Quickstart

### 1. Create the environment

The two documented constraints from [supplementary/README.md](supplementary/README.md) are: **Python ≤ 3.12** and **`faiss-cpu` from the `pytorch` conda channel**. Everything else is inferred from the imports across the code.

Conda (recommended, matches the original `torch-env`):

```bash
conda env create -f environment.yml      # creates `torch-env`
conda activate torch-env
```

Pip-only alternative (only if you can't use conda). Use Python 3.10–3.12; the PyPI `faiss-cpu` wheel is not published for 3.13+:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Smoke test the compressor core

From the repo root (so that `compressor.py` is on `sys.path`):

```bash
python - <<'PY'
import numpy as np
from compressor import Compressor, multi_exponents, all_moments

rng = np.random.default_rng(0)
data = rng.standard_normal((2000, 3))         # d=2000 points in R^3
cp = Compressor(data, random_state=0)
c, w = cp.compress(k=2, dstop=None)           # matches 0th–2nd moments
print("kept", c.size, "of", data.shape[0], "points;  sum(c) =", c.sum())

exps = multi_exponents(3, 2)
orig = sum(all_moments(x, exps) for x in data)
comp = sum(cj * all_moments(wj, exps) for cj, wj in zip(c, w))
print("max moment error:", float(np.max(np.abs(orig - comp))))
PY
```

Expected: `kept 10 of 2000` and a moment error near `1e-10` or smaller. 10 = `C(3+2, 2)`, the Tchakaloff bound.

### 3. Visual demo

```bash
python new/demo.py
```

Pops a side-by-side scatter of 1000 3-D Gaussian points and their 40-atom moment-matching compression (marker size ∝ weight).

### 4. Reproducing a figure

The per-figure mapping is in [PAPER_SUMMARY.md](PAPER_SUMMARY.md). The quickest runs:

- **Fig. 6 runtime** (minutes on a laptop): `cd runtime_benchmark && python runtime.py` → `time_list.csv`, then open `runtime_plot.ipynb`.
- **Fig. 2 error scaling** (dial down `d_list`, `mlist`, `k_list`, and `trials_per_d` at the bottom of `error_scaling/error_scaling.py` for a fast preview): `cd error_scaling && python error_scaling.py`.
- **Fig. 3 compressed training set** (minutes on a CPU): `cd new && python trainds.py` after creating `new/CPTDS/`.
- **Fig. 4 dynamical LTH** (many minutes, faster on a GPU): `cd new && python compress_dynamics.py`.
- **Figs. 5, 7** are Slurm-array sweeps over `(d, seed)`; see the `sid = int(os.environ["SLURM_ARRAY_TASK_ID"])` pattern in [scaling_law/trainds.py](scaling_law/trainds.py) and [scaling_law/width.py](scaling_law/width.py). For a single-seed preview, replace the `sid` block with hard-coded `seed, d`.

Every experiment folder has a `plot.ipynb` that reads the committed CSVs and emits the figure PDF.

### 5. Dependencies at a glance

What each library is used for:

| Library                 | Why it's needed                                                                                                                                                                   |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `numpy`                 | Everything.                                                                                                                                                                       |
| `scipy`                 | `scipy.special.jv` (cylindrical-harmonic target in Figs. 4, 5); `scipy.optimize.curve_fit` in the Fig. 5 plotting notebook.                                                       |
| `scikit-learn`          | `MiniBatchKMeans` clustering inside `Compressor.compress` when `alive ≫ dstop`.                                                                                                   |
| `joblib`                | Parallel `_reduce_compute` over clusters.                                                                                                                                         |
| `faiss-cpu`             | `IndexFlatL2` / `IndexIDMap2` for greedy nearest-neighbour subset search.                                                                                                         |
| `torch` (≥ 2.3)         | All training experiments. ≥ 2.3 is required because `scaling_law/width.py` imports `torch.amp.{autocast, GradScaler}` and `common.py` touches `torch.backends.mps.deterministic`. |
| `matplotlib`            | Every figure; `new/common.py::make_canvas` sets `pdf.use14corefonts=True`, so no custom fonts are installed.                                                                      |
| `pandas`                | CSV I/O in the plotting notebooks (`error_scaling/plot.ipynb`, `scaling_law/plot.ipynb`).                                                                                         |
| `jupyter` / `ipykernel` | Running the `plot.ipynb` notebooks.                                                                                                                                               |
| `tqdm`                  | Only used by `gpu/compressor_gpu.py` (archived prototype); safe to skip if you don't open that file.                                                                              |

No `pandas` / `tqdm` imports exist in `compressor.py` itself — the core library runs on `numpy + scipy + scikit-learn + joblib + faiss-cpu`.

### 6. Troubleshooting

- **`ModuleNotFoundError: No module named 'compressor'`** — run from the repo root, or `pip install -e .` once a `pyproject.toml` exists (see Milestone 2 in [CLEANUP_PLAN.md](CLEANUP_PLAN.md)). For now the experiment scripts use `sys.path.insert(0, parent)` to find it.
- **`faiss` install fails on Python 3.13** — downgrade to 3.12 (conda wheels). 3.13 wheels are not yet published as of this writing.
- **`torch.backends.mps.deterministic` AttributeError** — your PyTorch is older than 2.1; upgrade to ≥ 2.3 to match the pinned environment.
- **Out-of-memory during `Compressor.compress` at high `k`** — the moment matrix is `C(m+k, k) × d`. Drop `k`, or set `dstop` larger so more work stays in the cheap k-means phase.
- **IVF fallback message `[fallback] IVF search failed ...`** — harmless; the code switches itself to `IndexFlatL2` and retries. IVF is only beneficial at `d ≳ 10⁷`.

---

This repository was initially written by Hong-Yi Wang (hywang@princeton.edu), and later reorganized using Claude Code.
