# Release-quality cleanup plan

Goal: turn the current research scratch tree into a repo that a reader of [ICLR2026_A_Universal_Compression_Theory/ICLRver.tex](ICLR2026_A_Universal_Compression_Theory/ICLRver.tex) can clone and use to reproduce every figure and reuse the `Compressor` for their own work.

Companion: [PAPER_SUMMARY.md](PAPER_SUMMARY.md) (canonical figure → script mapping).

---

## Goals

**G1. One entry-point per figure.** Every figure in the paper (2–8) is produced by exactly one script or notebook, named for the figure it makes, with deterministic seeds and no environment-specific paths.

**G2. One canonical source for each piece of code.** The three drifted copies of `common.py`, `data_gen.py`, `compressor.py`, and the partially duplicated `compress_dynamics.py` / `trainds.py` collapse to a single authoritative copy under `moment_compression/` as an importable package.

**G3. Obvious reading order.** A reader lands on `README.md`, sees what the paper is about, how to install, which script makes which figure, and how to use `Compressor` on their own data — without opening more than one file.

**G4. Reproducibility preserved but not demanded.** Raw CSVs / per-seed `.pdf`s stay in the repo (so no one is forced to re-run a 2000-epoch Slurm sweep), but every one of them is regenerable by a command written at the top of the producing script.

**G5. Dead code removed.** Abandoned variants (`fxy/`, `gpu/`, `grok/`, `sine_regression/`, `MNIST/`, stray root-level artefacts like `job.slurm`, `output.txt`, `Compression_poster.pdf`, `data prelim.nb`, `ICLR 2026 reply.md`) are either deleted or moved into a clearly marked `attic/` that README calls out as not part of the paper.

**G6. Tests for the compression core.** `Compressor` has a small pytest suite covering: moment preservation to numerical tolerance, shape/weight invariants, k=0 trivial case, degenerate inputs (`d ≤ N_{m,k}`), and the k-means → greedy method switch.

**G7. License + citation surface.** The existing MIT `LICENSE` stays; `README.md` adds a BibTeX block matching the paper's author list and the arXiv/OpenReview link, so downstream users cite correctly.

---

## Target layout

```
moment_compression/                 # importable package
  __init__.py                       # re-exports Compressor, helpers
  compressor.py                     # the single core file
  symmetric.py                      # multi_exponents, all_moments, find_null_vec
  nn.py                             # TwoLayerNet, WeightedTwoLayerNet, compress_nn
  mha.py                            # MultiHeadAttention(+W), compress_mha
  data.py                           # generate_data, cyl_harmonic, generators
  plotting.py                       # make_canvas + rcParams

experiments/
  fig2_error_scaling/               # run.py + plot.ipynb + data/*.csv
  fig3_compress_trainds/
  fig4_lth/
  fig5_scaling_law/
  fig6_runtime/
  fig7_attention/
  fig8_polylog/

figures/                            # only the final PDFs used by the paper
tests/
  test_compressor.py
examples/
  demo_2d.py  demo_3d.py            # the current new/demo.py, kept as a tutorial

README.md                           # install + one-line-per-figure table
PAPER_SUMMARY.md                    # already exists
CLEANUP_PLAN.md                     # this file
LICENSE
pyproject.toml                      # pins numpy, faiss-cpu, torch, sklearn, joblib
ICLR2026_A_Universal_Compression_Theory/
attic/                              # fxy/, gpu/, grok/, sine_regression/, MNIST/, playground.ipynb
```

---

## Milestones

Each milestone ends with a **Pass criteria** gate. No milestone depends on a later one. Inside a milestone the work is broken down by the files it touches. Milestones are intentionally small so you can pause after any of them and still have a repo in a better state than before.

### Milestone 1 — Clean the working tree (half day)

**Objective.** Remove dead files and commit the pre-existing in-flight deletions so the starting point is consistent.

**File schedule.**

1. Decide the fate of currently-deleted-but-unstaged files: [Compression_poster.pdf](Compression_poster.pdf), [ICLR 2026 reply.md](ICLR%202026%20reply.md), [data prelim.nb](data%20prelim.nb), [job.slurm](job.slurm), [output.txt](output.txt). All are not cited from the paper — stage the deletes.
2. Move [playground.ipynb](playground.ipynb) → `attic/playground.ipynb`.
3. Move [fxy/](fxy/), [gpu/](gpu/), [grok/](grok/), [sine_regression/](sine_regression/), [MNIST/](MNIST/) → `attic/` and add a one-sentence `attic/README.md` explaining the category ("superseded or out-of-scope experiments").
4. Remove every tracked `.DS_Store`; add `.DS_Store`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/` to `.gitignore` (the repo currently has no `.gitignore`).
5. Remove the stale `TODO` list in [README.md](README.md).

**Pass criteria.**

- `git status` clean except for the single commit of M1.
- `find . -name .DS_Store -not -path "./.git/*"` returns nothing.
- `git ls-files | grep -E '(fxy|gpu|grok|sine_regression|MNIST)/'` returns only paths under `attic/`.

### Milestone 2 — Consolidate the core library (1–2 days)

**Objective.** Deduplicate the three copies of the supporting code and turn the root into an importable package, without changing any numeric output.

**File schedule.**

1. Create `moment_compression/__init__.py` re-exporting `Compressor`, `multi_exponents`, `all_moments`.
2. Move [compressor.py](compressor.py) → `moment_compression/compressor.py` unchanged. Split the free functions (`multi_exponents`, `all_moments`, `find_null_vec`) into `moment_compression/symmetric.py`; keep thin re-exports from `compressor.py` for backward compatibility.
3. Promote [new/common.py](new/common.py) as the canonical `moment_compression/nn.py` + `moment_compression/plotting.py` + `moment_compression/data.py`, chosen over [scaling_law/common.py](scaling_law/common.py), [supplementary/common.py](supplementary/common.py), [transformer/common.py](transformer/common.py) because (a) it is used by the headline figure (Fig. 4), (b) it has `compress_nn` that the others also need, (c) it has the newest `make_canvas`.
4. Diff the other three `common.py`s against the canonical one; port any unique pieces (e.g. the MHA-specific bits from [transformer/common.py](transformer/common.py) go into `moment_compression/mha.py` beside [transformer/mha.py](transformer/mha.py)).
5. Replace every `sys.path.append(..)` + `from compressor import Compressor` / `from common import ...` shim (there are 10+) with `from moment_compression import ...`.
6. Add `pyproject.toml` pinning `numpy`, `scipy`, `scikit-learn`, `joblib`, `faiss-cpu`, `torch` (optional), `matplotlib`; declare entry-point `moment_compression`.
7. Regenerate `playground.ipynb`-independent smoke runs: `python -c "from moment_compression import Compressor; ..."` must work from any directory.

**Pass criteria.**

- `pip install -e .` succeeds on a fresh venv.
- `python -c "import moment_compression; print(moment_compression.Compressor)"` works.
- `grep -R "sys.path.append" --include='*.py'` returns 0 hits outside of `attic/`.
- Running the Fig. 4 headline script (`experiments/fig4_lth/run.py`, see M3) produces bitwise-identical output to the pre-refactor [new/compress_dynamics.py](new/compress_dynamics.py) on seed 42.

### Milestone 3 — Re-home every figure pipeline (2–3 days)

**Objective.** Make one `experiments/figN_*` folder per paper figure, each containing `run.py` (script that recomputes the CSV), `plot.ipynb` or `plot.py` (makes the pdf), and a `data/` subfolder with the committed CSVs. Remove the old scattered locations.

**File schedule (figure by figure, same pattern each time).**

- Fig. 2: move [error_scaling/error_scaling.py](error_scaling/error_scaling.py) → `experiments/fig2_error_scaling/run.py`; move [error_scaling/plot.ipynb](error_scaling/plot.ipynb) → `experiments/fig2_error_scaling/plot.ipynb`; move `error_list.csv`, `sqrt_error_list.csv` → `experiments/fig2_error_scaling/data/`. Lift the hard-coded `mlist`/`klist`/`d_list` into a CLI (`argparse`) so M6's tests can drive a tiny version.
- Fig. 3: [new/trainds.py](new/trainds.py) → `experiments/fig3_compress_trainds/run.py`; prune the commented optimiser blocks; move `new/CPTDS/teacher_AdamW_*` CSVs used by the final plot into `experiments/fig3_compress_trainds/data/`; extract the plotting cells from [new/plot.ipynb](new/plot.ipynb) into `plot.ipynb` next door.
- Fig. 4: [new/compress_dynamics.py](new/compress_dynamics.py) → `experiments/fig4_lth/run.py` (CLI over optimiser/lr); `new/LTH/harm_*` CSVs used in the paper → `data/`.
- Fig. 5: [scaling_law/trainds.py](scaling_law/trainds.py) + [scaling_law/width.py](scaling_law/width.py) → `experiments/fig5_scaling_law/run_trainds.py` + `run_width.py`; [scaling_law/linreg.py](scaling_law/linreg.py) → `run_linreg.py`; all CSVs used for the fit go under `data/`; [scaling_law/plot.ipynb](scaling_law/plot.ipynb) stays alongside. Replace `SLURM_ARRAY_TASK_ID` parsing with `argparse --d --seed` (plus a thin `launch.slurm` that sets those from the array id).
- Fig. 6: [runtime_benchmark/\*](runtime_benchmark/) → `experiments/fig6_runtime/` straight copy.
- Fig. 7: [transformer/ICL_piecewise.py](transformer/ICL_piecewise.py) → `experiments/fig7_attention/run.py`; drop [transformer/ICL.py](transformer/ICL.py) / [transformer/compress_dynamics.py](transformer/compress_dynamics.py) / [transformer/teacher.py](transformer/teacher.py) into `attic/transformer_early/` (they are the pre-piecewise variants).
- Fig. 8: [error_scaling/polylog_error.py](error_scaling/polylog_error.py) → `experiments/fig8_polylog/run.py`; CSVs under `data/`; plotting cells copied out of [error_scaling/plot.ipynb](error_scaling/plot.ipynb).
- Final PDFs already in [figures/](figures/) stay; delete pdf/png files in [figures/](figures/) that are not referenced by `ICLRver.tex` (e.g. `mainidea.pdf`, `NN_grouped.pdf`, `mnist_16_*`).

**Pass criteria.**

- `ls experiments/` prints exactly `fig2_error_scaling fig3_compress_trainds fig4_lth fig5_scaling_law fig6_runtime fig7_attention fig8_polylog`.
- Each `experiments/figN_*/run.py` executes with a documented small-`d` "smoke" config (`python run.py --smoke`) in < 2 minutes on a laptop, and produces a non-empty csv under `data/`.
- Each `plot.ipynb` re-runs top-to-bottom against the committed `data/*.csv` and writes the same PDF that is in [figures/](figures/) (visual identity check in the ipynb output).
- `grep -R "from compressor import\|sys.path.append" experiments/` is empty.

### Milestone 4 — README + docstrings + examples (1 day)

**Objective.** A visitor can figure out what the repo is in 60 seconds.

**File schedule.**

1. Rewrite [README.md](README.md): (i) one-paragraph description with arXiv/OpenReview link, (ii) install block, (iii) 60-second example (`Compressor` on 2D Gaussian, 5 lines), (iv) table mapping fig2..fig8 to `experiments/figN_*/run.py`, (v) citation BibTeX.
2. Add module-level docstrings to every file under `moment_compression/` (one paragraph: what it does, which theorem/algorithm it implements, which figure uses it).
3. Move `new/demo.py` → `examples/demo_2d_3d.py`; remove the `print("Lattice+FPS min dist:", "8")` placeholder in the module under `data.py`.
4. Add a `REPRODUCING.md` with per-figure compute requirements (rough wall-clock on a stated laptop / GPU).

**Pass criteria.**

- `markdown-link-check README.md` passes (no broken relative links).
- Every `.py` under `moment_compression/` has a module docstring; `pydocstyle moment_compression/ --select=D100` is clean.
- `python examples/demo_2d_3d.py` runs end-to-end and pops a two-panel plot.

### Milestone 5 — Tests (half day)

**Objective.** Protect the core algorithm from silent regressions when future readers extend it.

**File schedule.** Add `tests/test_compressor.py` covering:

1. `compress(k=k, dstop=N_{m,k})` on `d=10000` Gaussian points preserves every order-≤k moment to < 1e-8.
2. Weighted input reproduces same moments with scaling: feeding `(data, weights=c)` then compressing should match feeding `(repeated data)` within tolerance.
3. `d ≤ N_{m,k}` returns input unchanged.
4. Method-switch: setting `greedy_threshold` forces the k-means path on a size that would have gone greedy, same output.
5. Deterministic given `random_state=0` across two fresh runs.

Add `.github/workflows/ci.yml` that installs with pip and runs pytest (optional; gated on the author's willingness to maintain CI).

**Pass criteria.**

- `pytest tests/` green on the author's laptop in < 60s.
- Mutating `_reduce_compute` to drop the final `c[c < tol] = 0.0` line makes test 1 fail — i.e. the test is actually load-bearing.

### Milestone 6 — Final audit (half day)

**Objective.** Make sure the three key promises of the reproducibility statement (Sec. "Reproducibility statement", p. 10) hold.

**File schedule.**

1. For each of Fig. 2–8, run the `--smoke` config and confirm the csv + plot land in the right place.
2. Confirm `git clone … && pip install -e . && pytest` works on a fresh venv.
3. Open every `.py` in `experiments/` and confirm top-of-file header: one-line figure description, reference to the paper section, command to run.
4. Update `PAPER_SUMMARY.md` links to the new `experiments/figN_*` paths.
5. Tag `v1.0.0-iclr2026`.

**Pass criteria.**

- The "Reproducibility statement" URL in the paper (currently https://github.com/WHY-David/moment_compression.git) points at a repo whose `README.md` lists seven figures, seven run scripts, seven plotting notebooks, and a working `Compressor` demo.
- A friend (simulated: pick a non-author) can reproduce Fig. 2 (smoke) and Fig. 6 end-to-end in under an hour without asking a question.
