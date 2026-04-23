# Release-quality cleanup plan

Goal: turn the current research scratch tree into a repo that a reader of [ICLR2026_A_Universal_Compression_Theory/ICLRver.tex](ICLR2026_A_Universal_Compression_Theory/ICLRver.tex) can clone and use to reproduce every figure and reuse the `Compressor` for their own work.

Companion: [PAPER_SUMMARY.md](PAPER_SUMMARY.md) (canonical figure → script mapping).

---

## Goals

**G1. One folder per figure.** Every figure in the paper (2–8) lives in its own top-level folder (`fig2_error_scaling/`, …, `fig8_polylog/`) that contains the run script(s), the plotting notebook, and the CSV data. In addition, one top-level `demo/` folder shows the basic use cases of `Compressor` — it is essentially a renamed [supplementary/](supplementary/).

**G2. One canonical `compressor.py`.** The duplicated [supplementary/compressor.py](supplementary/compressor.py) is removed; every script imports the single root-level [compressor.py](compressor.py). The per-folder `common.py` and `data_gen.py` files are intentionally **not** merged — each one has diverged slightly for its own experiment and consolidating them would cause silent behaviour changes.

**G3. Obvious reading order.** A reader lands on `README.md`, sees what the paper is about, how to install, which folder makes which figure, and how to use [compressor.py](compressor.py) on their own data — without opening more than one file.

**G4. Data stays in its folder.** Raw CSVs and per-seed `.pdf`s stay committed where the producing script wrote them, so nobody is forced to re-run a 2000-epoch Slurm sweep to see a plot.

**G5. Unused code audit (review-before-delete).** Produce an itemised list of files that no script, notebook, or figure references. Hand the list to the author for review. **Do not delete anything in this step.**

**G6. License + citation surface.** The existing MIT `LICENSE` stays; `README.md` adds an OpenReview link and a BibTeX block for the arXiv preprint:

- OpenReview forum: <https://openreview.net/forum?id=vxkzW4ljeX&nesting=2&sort=date-desc>
- arXiv preprint: <https://arxiv.org/abs/2510.00504>

  ```bibtex
  @misc{wang2025universalcompression,
    title        = {A Universal Compression Theory for the Lottery Ticket Hypothesis and Neural Scaling Laws},
    author       = {Hong-Yi Wang and Di Luo and Tomaso Poggio and Isaac L. Chuang and Liu Ziyin},
    year         = {2025},
    eprint       = {2510.00504},
    archivePrefix= {arXiv},
    primaryClass = {cs.LG},
    url          = {https://arxiv.org/abs/2510.00504}
  }
  ```

---

## Target layout

```
compressor.py                       # the single shared core; README shows how to use it

fig2_error_scaling/                 # run script(s) + plot.ipynb + the CSVs used to make the final pdf
fig3_compress_trainds/
fig4_lth/
fig5_scaling_law/
fig6_runtime/
fig7_attention/
fig8_polylog/

demo/                               # basic use cases of Compressor (renamed from supplementary/)

figures/                            # only the final PDFs referenced by the paper

README.md                           # install + one-line-per-figure table + how to use compressor.py + citations
PAPER_SUMMARY.md                    # figure → script map
CLEANUP_PLAN.md                     # this file
LICENSE
requirements.txt / environment.yml  # existing pins
ICLR2026_A_Universal_Compression_Theory/
```

---

## Milestones

Each milestone ends with a **Pass criteria** gate. No milestone depends on a later one. Inside a milestone the work is broken down by the files it touches. Milestones are intentionally small so you can pause after any of them and still have a repo in a better state than before.

### Milestone 1 — Produce a "candidate-unused" list (half day)

**Objective.** Enumerate every file that no script, notebook, or figure references, and hand the list to the author. **Do not delete anything in this milestone.**

**Procedure.**

1. Build the reference set: union of (i) every file path mentioned in [PAPER_SUMMARY.md](PAPER_SUMMARY.md), (ii) every `include…{}` / `\includegraphics{}` in [ICLR2026_A_Universal_Compression_Theory/ICLRver.tex](ICLR2026_A_Universal_Compression_Theory/ICLRver.tex), (iii) every file imported / opened by scripts in the reference set (transitive closure).
2. List every file NOT in the reference set (excluding `.git/`, `__pycache__/`, `.DS_Store`, `.ipynb_checkpoints/`).
3. Group the list by directory and annotate each item with _why_ it looks unused (e.g. "earlier draft of ICL", "CSV with unused `lr=0.001` config", "figure not in the paper").

**Initial candidate list** (to be extended during the milestone, then reviewed by the author):

- Root: [CLEANUP_PLAN.md](CLEANUP_PLAN.md) is kept for authors only — decide whether to ship it.
- `error_scaling/`: [error_scaling/error_uniform.csv](error_scaling/error_uniform.csv) (not referenced by any script or the plot notebook).
- `new/`: [new/trainds_linear.py](new/trainds_linear.py) (not referenced; appears to be a linear-model prototype that never made it into Fig. 3).
- `scaling_law/`: [scaling_law/trainds_w_epoch.py](scaling_law/trainds_w_epoch.py) (not referenced; epoch-varying variant not used in the final Fig. 5).
- `transformer/`: [transformer/ICL.py](transformer/ICL.py), [transformer/compress_dynamics.py](transformer/compress_dynamics.py), [transformer/teacher.py](transformer/teacher.py) — flagged in `PAPER_SUMMARY.md` as "earlier draft"; verify they are not imported before deletion. Also the older loss CSVs/PDFs whose hyperparameters do not match the filename loaded in [transformer/plot.ipynb](transformer/plot.ipynb) (namely `lr0.0001_ep50_spe5`): `icl_piecewise_losses.csv`, `icl_piecewise_losses.pdf`, `icl_piecewise_losses_d8000_dstop800_lr0.0002_ep50*`, `…lr0.001_ep50*`. Local `transformer/attention.pdf` is the copy used for panel assembly — check whether it duplicates `figures/attention.pdf` before deleting.
- `figures/`: `NN_grouped.pdf`, `NN_init.pdf`, `NN_trained.pdf`, `mainidea.pdf`, `mnist_16_opacity.png`, `mnist_16_tiled.png` — none are `\includegraphics`'d in the final `ICLRver.tex` (the `mainidea.pdf` reference is commented out).

**Housekeeping that is not review-gated and can happen here:**

- Add `.gitignore` entries for `.DS_Store`, `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/` (already done for `__pycache__/`, `.claude/`, `.vscode/`).
- Remove the stale `TODO` list at the top of [README.md](README.md).

**Pass criteria.**

- A file `UNUSED_CANDIDATES.md` (or inline section in this file) exists with the full candidate list, annotated. The author has signed off on what to delete; no deletions have been made yet.

### Milestone 2 — Remove the duplicate `compressor.py` (half day)

**Objective.** A single authoritative [compressor.py](compressor.py) at the repo root. The per-folder `common.py` and `data_gen.py` files stay in place — they have diverged per experiment and merging them would risk silent numerical changes.

**File schedule.**

1. Diff [supplementary/compressor.py](supplementary/compressor.py) against [compressor.py](compressor.py); confirm the root version is a strict superset (or that any unique bits in `supplementary/` are dead). Delete [supplementary/compressor.py](supplementary/compressor.py).
2. Update every `from compressor import ...` in [supplementary/](supplementary/) (soon to be `demo/`, see M3) to reach the root `compressor.py` — either by `sys.path` append with a comment, or by running those scripts from the repo root.
3. Verify no other folder has a private copy of `compressor.py`: `find . -name compressor.py -not -path './.git/*'` should list exactly one file.
4. Leave [error_scaling/data_gen.py](error_scaling/data_gen.py), [new/data_gen.py](new/data_gen.py), [runtime_benchmark/data_gen.py](runtime_benchmark/data_gen.py), [scaling_law/data_gen.py](scaling_law/data_gen.py) untouched; same for the four `common.py`s ([new/common.py](new/common.py), [scaling_law/common.py](scaling_law/common.py), [transformer/common.py](transformer/common.py), [supplementary/common.py](supplementary/common.py)).

**Pass criteria.**

- `find . -name compressor.py -not -path './.git/*'` returns only `./compressor.py`.
- Every figure's plot notebook and run script still produces the same CSV/PDF byte-for-byte as before M2 on a fixed seed (no numeric drift from the dedupe).

### Milestone 3 — One folder per figure + `demo/` (2–3 days)

**Objective.** Flat top-level layout: one `figN_*/` folder per paper figure (run script + plot notebook + the CSVs used for the final PDF), plus a `demo/` folder for basic `Compressor` use cases. Remove the old scattered locations.

**File schedule (figure by figure).**

- Fig. 2 (`fig2_error_scaling/`): rename/move [error_scaling/](error_scaling/) → `fig2_error_scaling/`. Keep [error_scaling.py](error_scaling/error_scaling.py), [data_gen.py](error_scaling/data_gen.py), [job_array.py](error_scaling/job_array.py), [plot.ipynb](error_scaling/plot.ipynb), and the CSVs `error_list.csv` / `sqrt_error_list.csv` in that folder. The Fig. 8 pieces ([polylog_error.py](error_scaling/polylog_error.py), `polylog_error_1d.csv`, `polylog_error_2d.csv`) split off into `fig8_polylog/` (below).
- Fig. 3 (`fig3_compress_trainds/`): from [new/](new/) move [trainds.py](new/trainds.py), [trainds_fullbatch.py](new/trainds_fullbatch.py), [data_gen.py](new/data_gen.py), the CPTDS CSVs actually used by [new/plot.ipynb](new/plot.ipynb), and the Fig. 3 cells of that notebook.
- Fig. 4 (`fig4_lth/`): from [new/](new/) move [compress_dynamics.py](new/compress_dynamics.py), [common.py](new/common.py), the `LTH/harm_*` CSVs used by the panel, and the Fig. 4 cells of [new/plot.ipynb](new/plot.ipynb).

  Note: Fig. 3 and Fig. 4 currently share `new/common.py`, `new/data_gen.py`, and `new/plot.ipynb`. Since we agreed not to merge `common.py` / `data_gen.py`, duplicate them into both folders rather than hoisting to a shared location. Split the notebook into one per figure.

- Fig. 5 (`fig5_scaling_law/`): rename/move [scaling_law/](scaling_law/) → `fig5_scaling_law/`. Keep [trainds.py](scaling_law/trainds.py), [width.py](scaling_law/width.py), [common.py](scaling_law/common.py), [data_gen.py](scaling_law/data_gen.py), [plot.ipynb](scaling_law/plot.ipynb), [trainds_scaling.csv](scaling_law/trainds_scaling.csv), and the `LTH_harm_AdamW_k6_trains1000000_…/` data folder referenced by the width panel.
- Fig. 6 (`fig6_runtime/`): rename/move [runtime_benchmark/](runtime_benchmark/) → `fig6_runtime/` unchanged.
- Fig. 7 (`fig7_attention/`): rename/move [transformer/](transformer/) → `fig7_attention/`. Keep [mha.py](transformer/mha.py), [ICL_piecewise.py](transformer/ICL_piecewise.py), [common.py](transformer/common.py), [plot.ipynb](transformer/plot.ipynb), and the one matched-hyperparameter loss CSV loaded by the notebook. The earlier-draft scripts ([ICL.py](transformer/ICL.py), [compress_dynamics.py](transformer/compress_dynamics.py), [teacher.py](transformer/teacher.py)) and the non-matching loss CSVs are on the G5 candidate list (M1) — the author decides their fate before M3 starts.
- Fig. 8 (`fig8_polylog/`): from `fig2_error_scaling/` after the Fig. 2 move, split out [polylog_error.py](error_scaling/polylog_error.py), `polylog_error_1d.csv`, `polylog_error_2d.csv`, and the Fig. 8 cells of the plot notebook.
- `demo/`: rename [supplementary/](supplementary/) → `demo/`. After M2 this folder contains `common.py`, `compress_dynamics.py`, `data_gen.py`, `demo.py`, `trainds.py`, `README.md` — it's a self-contained "basic use cases of `Compressor`" tutorial. Delete [supplementary/compressor.py](supplementary/compressor.py) (done in M2) and update the `README.md` so it points at the root-level [compressor.py](compressor.py).
- [figures/](figures/): keep only PDFs actually `\includegraphics`'d in [ICLRver.tex](ICLR2026_A_Universal_Compression_Theory/ICLRver.tex); the unused PDFs/PNGs identified in M1 are removed after author sign-off.

**Pass criteria.**

- `ls` at the repo root shows exactly the seven `figN_*` folders + `demo/` + `figures/` + `compressor.py` + the three docs + `LICENSE` + env files + the tex directory.
- Each `figN_*/` folder contains a `plot.ipynb` that re-runs top-to-bottom against the committed CSVs and writes the PDF that is in [figures/](figures/).
- `grep -R "from compressor import\|sys.path.append" fig?_*/ demo/` either resolves cleanly to the root [compressor.py](compressor.py) or returns no hits.

### Milestone 4 — README (1 day)

**Objective.** A visitor can figure out what the repo is in 60 seconds, and knows how to use [compressor.py](compressor.py) on their own data.

**File schedule.** Rewrite [README.md](README.md) to include:

1. **One-paragraph description** of the paper, with links to OpenReview and arXiv.
2. **Install block** (`pip install -r requirements.txt` or the `environment.yml` conda recipe).
3. **How to use `compressor.py`.** A minimal worked example that loads the single root-level [compressor.py](compressor.py), runs `Compressor` on, say, a 2D point cloud, and shows how the output weights reproduce the first `k` moments. Cover:
   - What `Compressor(data, weights=…)` expects (shape, types).
   - The key knobs: `k`, `dstop`, and the k-means ↔ greedy switch.
   - How to read the compressed output (the `(support, weights)` pair).
   - Where to look inside [compressor.py](compressor.py) for `_find_best_subset` and `_reduce_compute` if the reader wants to extend the algorithm.
4. **Figure table** mapping `fig2_error_scaling/` … `fig8_polylog/` to the run script(s) and plot notebook inside each folder, plus a line for `demo/`.
5. **Citations section**, with the two items from G6:
   - OpenReview forum: <https://openreview.net/forum?id=vxkzW4ljeX&nesting=2&sort=date-desc>
   - arXiv preprint: <https://arxiv.org/abs/2510.00504> followed by the BibTeX block shown in G6.

Also: remove the stale `TODO` list at the top of the current [README.md](README.md) (already listed under M1 housekeeping).

**Pass criteria.**

- `README.md` renders with no broken relative links.
- The "How to use `compressor.py`" snippet copy-pastes and runs against the root-level [compressor.py](compressor.py) with only the pinned deps installed.
- The OpenReview URL and the BibTeX entry match exactly the strings in G6.

### Milestone 5 — Final audit (half day)

**Objective.** Make sure the reproducibility statement (Sec. "Reproducibility statement", p. 10) holds for the new layout.

**File schedule.**

1. For each of Fig. 2–8, re-run the plot notebook against the committed CSVs and visually compare to the PDF in [figures/](figures/).
2. Open every `.py` in `figN_*/` and `demo/` and confirm a top-of-file header: one-line figure description, reference to the paper section, command to run.
3. Update [PAPER_SUMMARY.md](PAPER_SUMMARY.md) links to the new `figN_*/` paths.
4. Tag `v1.0.0-iclr2026`.

**Pass criteria.**

- The "Reproducibility statement" URL in the paper points at a repo whose `README.md` lists seven figures, seven run scripts, seven plotting notebooks, a `demo/` with a working `Compressor` example, and a BibTeX block.
- A non-author can reproduce Fig. 2 and Fig. 6 end-to-end in under an hour without asking a question.

---

## M1 audit: candidate-unused files (2026-04-23)

Produced per G5. **Nothing in this list has been deleted.** Author signs off before deletion.

Method: the reference set is the union of (i) files explicitly cited in [PAPER_SUMMARY.md](PAPER_SUMMARY.md); (ii) `\includegraphics{…}` entries in [ICLRver.tex](ICLR2026_A_Universal_Compression_Theory/ICLRver.tex) (ignoring commented-out blocks — `mainidea.pdf` is commented out at line 217); (iii) filenames constructed dynamically inside the used plot notebooks and run scripts.

| File / group                                                                                                                                                     | Why it looks unused                                                                                                                                                                                                                         |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [error_scaling/error_uniform.csv](error_scaling/error_uniform.csv)                                                                                               | Not referenced by `error_scaling.py`, `polylog_error.py`, or `plot.ipynb`.                                                                                                                                                                  |
| [new/trainds_linear.py](new/trainds_linear.py)                                                                                                                   | Not referenced anywhere; linear-model prototype that did not make it into Fig. 3.                                                                                                                                                           |
| [scaling_law/trainds_w_epoch.py](scaling_law/trainds_w_epoch.py)                                                                                                 | Not referenced anywhere; epoch-varying variant not used in the final Fig. 5.                                                                                                                                                                |
| [transformer/ICL.py](transformer/ICL.py), [transformer/compress_dynamics.py](transformer/compress_dynamics.py), [transformer/teacher.py](transformer/teacher.py) | Flagged as "earlier draft" in [PAPER_SUMMARY.md](PAPER_SUMMARY.md) (pre-piecewise ICL variant). Not imported by [transformer/ICL_piecewise.py](transformer/ICL_piecewise.py) or [transformer/plot.ipynb](transformer/plot.ipynb).           |
| [transformer/icl_piecewise_losses.csv](transformer/icl_piecewise_losses.csv), [transformer/icl_piecewise_losses.pdf](transformer/icl_piecewise_losses.pdf)       | `plot.ipynb` reads only `icl_piecewise_losses_d8000_dstop800_lr0.0001_ep50_spe5.csv`. These bare filenames are older runs.                                                                                                                  |
| `transformer/icl_piecewise_losses_d8000_dstop800_lr0.0002_ep50{,.pdf,_spe2.csv}`, `…lr0.001_ep50.{csv,pdf}`                                                      | Different hyperparameters than the one loaded by `plot.ipynb`; superseded.                                                                                                                                                                  |
| [transformer/attention.pdf](transformer/attention.pdf)                                                                                                           | Byte-identical to [figures/attention.pdf](figures/attention.pdf) (verified with `diff`). Only the `figures/` copy is `\includegraphics`'d by the paper.                                                                                     |
| [figures/NN_grouped.pdf](figures/NN_grouped.pdf), [figures/NN_init.pdf](figures/NN_init.pdf), [figures/NN_trained.pdf](figures/NN_trained.pdf)                   | Not `\includegraphics`'d in `ICLRver.tex`.                                                                                                                                                                                                  |
| [figures/mainidea.pdf](figures/mainidea.pdf)                                                                                                                     | Only reference in `ICLRver.tex` is a commented-out block (lines 217–219).                                                                                                                                                                   |
| [figures/mnist_16_opacity.png](figures/mnist_16_opacity.png), [figures/mnist_16_tiled.png](figures/mnist_16_tiled.png)                                           | Not `\includegraphics`'d in `ICLRver.tex`; leftover from the dropped MNIST exploration.                                                                                                                                                     |
| [figures/NSL_dataset.pdf](figures/NSL_dataset.pdf)                                                                                                               | The paper `\includegraphics`'s `NSL.pdf` (a combined panel). `NSL_dataset.pdf` / `NSL_width.pdf` are per-panel PDFs generated by the plot notebook; kept inside `scaling_law/` but the `figures/` copy of `NSL_dataset.pdf` is a duplicate. |

Notes / watch-outs (not in the delete list, but flagged for the author):

- [scaling_law/plot.ipynb](scaling_law/plot.ipynb) defaults point at `CPU_harm_AdamW_k6_trains20000_…_epoch2000/`, which the author already deleted (git shows the deletion pending). The folder that survived on disk is `LTH_harm_AdamW_k6_trains1000000_…_epoch200/`. The notebook defaults need updating before Fig. 5 is reproducible — but that is a code fix, not a delete-candidate.
- The following files were **already deleted by the author** but not yet staged (git status `D`): `new/data.csv`, `new/demo.py`, `scaling_law/linreg.py`, `scaling_law/CPTDS/*`, `scaling_law/CPU_harm_…_epoch2000/*`, `scaling_law/LTH_harm_AdamW_k6_noise0.2_bs512_lr0.001/*`, `gpu/*`. M3 will stage these along with the moves.
- `error_scaling/error_scaling.py` writes `sqrt_error_list.csv`; that CSV is not currently on disk but will be re-generated by a run. Leave the path reference in the script as-is.

**Action required from author: reply with "delete all" or an itemised list of what to keep before M3 executes the removal.**
