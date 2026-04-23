# Paper summary and figure-to-code map

Paper: **A Universal Compression Theory for the Lottery Ticket Hypothesis and Neural Scaling Laws** (Wang, Luo, Poggio, Chuang, Ziyin; ICLR 2026 submission).
Source: [ICLR2026_A_Universal_Compression_Theory/ICLRver.tex](ICLR2026_A_Universal_Compression_Theory/ICLRver.tex).

---

## 1. What the paper proves

Every machine-learning problem with a permutation-symmetric "object" set (data points, neurons, attention heads, ensemble members) can be losslessly compressed, in the asymptotic sense, from `d` objects to `d' = O(polylog d)` weighted objects. The compression is realised by matching the first `k` statistical moments of the object distribution.

Main theoretical results:

- **Multivariate FTSP** (Theorem in `sec: theory`). Any symmetric polynomial of `{w_i} ⊂ R^m` is a function only of the tensor moments `p_k = (1/d) Σ w_i^{⊗k}`.
- **Tchakaloff bound.** `N_{m,k} = C(m+k, k)` weighted objects suffice to reproduce moments up to order `k`.
- **Universal Compression theorem (Thm. `compression_errorbound`).** Algorithm 1 achieves error `|φ(c') − φ(c)| = O(d · d'^{1−(k+1)/m})`; for `k > m−1` this gives `d' = O((d/ε)^{m/(k−m+1)})`.
- **polylog-optimality (Thm. `errorbound_polylog`).** The best choice `k_opt = Θ(d'^{1/m})` gives compression `d → log^m(d/ε)`; a matching lower bound shows this rate is optimal up to constants.
- **Dynamical LTH (Thm. `DLTH`).** Because standard update rules commute with permutations, applying the theorem with `f ∘ T` (prediction composed with training) yields a compressed _initial_ network whose trained prediction matches the original throughout training.
- **Improved neural scaling laws.** Compressing `d → d^σ` turns `L ∝ d^{-α}` into `L ∝ d'^{-α/σ}`; compressing to `log^m d` upgrades any power law to a stretched exponential.

## 2. The algorithm (what the code must implement)

Algorithm family `alg:general` (Sec. 3.2) and concrete algorithm `alg:reduce` (Appendix `app:algorithm`):

1. **Cluster step.** Find a subset `S ⊆ supp(c)` with `|S| > N_{m,k}` and diameter `O(|supp c|^{−1/m})`. Exact NN-minimum-diameter cluster is NP-hard; implementation uses a two-stage strategy:
   - **k-means / MiniBatchKMeans** (sample-weighted) while `|supp c| ≫ d'`: parallel reduce each cluster.
   - **Greedy FAISS nearest-neighbour search** (flat `IndexFlatL2`, fallback from IVF) once `|supp c|` approaches `d'`.
2. **Moment-match reduce step.** For each cluster form the moment matrix `A ∈ R^{N_{m,k} × |S|}` whose columns are `w_i^{⊗≤k}`, then Carathéodory-peel: repeatedly find `z ∈ ker(A_S)`, step `c_S ← c_S − t z` with `t = min_{z_i>0} c_i/z_i`, dropping any `c_i` that hits zero. Null vectors come from SVD, or a faster ridge-regularised random probe when `|S| ≫ N_{m,k}`.

Object-type-specific wrappers:

- **Dataset compression.** Each row `w_i = (x_i, y_i)` is a data point; compressed weights feed a `WeightedRandomSampler` as the mini-batch distribution.
- **Two-layer NN width compression.** Each neuron contributes `w_i = [v_i, W_{1,i}, b_{1,i}]`. After compression the second-layer weights absorb `c_j`, and gradients are rescaled by `1/c_j` on the hidden row to implement the "compressed dynamics" `T'`.
- **Multi-head attention.** Each head carries `(W_q^{(h)}, W_k^{(h)}, W_v^{(h)}, W_o^{(h)})`; compression keeps `d'` heads with head-weights, gradients rescaled analogously.

## 3. Figures and code

Figure 1 (`illustration.pdf`) is a schematic and is excluded. Each entry below lists the figure, its location in the PDF, and the script(s) + data files that generate it.

### Fig. 2 — Error scaling for a generic symmetric function (`error_scaling.pdf`, Sec. 3.2)

Tests `|f(θ') − f(θ)| ∝ d^{-α}` for the sigmoid-of-inner-product function in Eq. (5), varying `m ∈ {1..5}` and `k`.

| Step                     | File                                                                                                                                                                                                |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data / compression sweep | [error_scaling/error_scaling.py](error_scaling/error_scaling.py) (hard-coded grid over `m, k, d`; appends to `sqrt_error_list.csv` / `error_list.csv`)                                              |
| Random point generator   | [error_scaling/data_gen.py](error_scaling/data_gen.py)                                                                                                                                              |
| Slurm job launcher       | [error_scaling/job_array.py](error_scaling/job_array.py)                                                                                                                                            |
| Panels (a–e) plot        | [error_scaling/plot.ipynb](error_scaling/plot.ipynb) using `error_list.csv` / `sqrt_error_list.csv`; writes `errors_m{1..5}.pdf`, `error_alpha.pdf`, then combined into `figures/error_scaling.pdf` |

### Fig. 3 — Compression of the training dataset (`compress_trainds.pdf`, Sec. 3.2)

Teacher–student setup, 2-layer ReLU MLP teacher; compare training on full `d=10^4`, naïve subsample `d'=10^3`, and moment-matched `d'=10^3`.

| Step                                                       | File                                                                                               |
| ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| Dataset generator                                          | [new/data_gen.py](new/data_gen.py)                                                                 |
| Training + plot (full-batch variant)                       | [new/trainds_fullbatch.py](new/trainds_fullbatch.py)                                               |
| Training + plot (mini-batch variant, used in final panels) | [new/trainds.py](new/trainds.py) writing into [new/CPTDS/](new/CPTDS/)                             |
| Panel composition                                          | [new/plot.ipynb](new/plot.ipynb) (assembles panels a–d and exports `figures/compress_trainds.pdf`) |

### Fig. 4 — Dynamical lottery ticket hypothesis (`LTH.pdf`, Sec. 4)

2-layer ReLU NN of width `d=10^4` vs. compressed width `d'=10^3` with `k=5`, across SGD / Adam / RMSprop / etc., learning the cylindrical harmonic `J_6(20r)cos(6θ)`.

| Step                                  | File                                                                                                                        |
| ------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| Data + NN + compression helpers       | [new/common.py](new/common.py) (defines `TwoLayerNet`, `WeightedTwoLayerNet`, `compress_nn`, `cyl_harmonic`, `make_canvas`) |
| Per-optimiser training run            | [new/compress_dynamics.py](new/compress_dynamics.py) writing into [new/LTH/](new/LTH/)                                      |
| Panel (a) ground-truth image          | [new/plot.ipynb](new/plot.ipynb)                                                                                            |
| Panel composition → `figures/LTH.pdf` | [new/plot.ipynb](new/plot.ipynb)                                                                                            |

### Fig. 5 — Improved neural scaling law (`NSL.pdf`, Sec. 5)

Panel (a) uses the Fig. 3 task; panel (b) uses the Fig. 4 task. For both panels `d' = [16√d]` with `k=6`.

| Step                                                  | File                                                                                                                                                                                                                        |
| ----------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| (a) Dataset-size scaling (Slurm array over `d, seed`) | [scaling_law/trainds.py](scaling_law/trainds.py) → [scaling_law/trainds_scaling.csv](scaling_law/trainds_scaling.csv)                                                                                                       |
| (b) Width scaling (Slurm array over `d, seed`)        | [scaling_law/width.py](scaling_law/width.py) → per-run csvs in [scaling_law/LTH_harm_AdamW_k6_trains1000000_noise0.2_bs512_lr0.001_epoch200/](scaling_law/LTH_harm_AdamW_k6_trains1000000_noise0.2_bs512_lr0.001_epoch200/) |
| Shared helpers (model, cyl harmonic, canvas)          | [scaling_law/common.py](scaling_law/common.py), [scaling_law/data_gen.py](scaling_law/data_gen.py)                                                                                                                          |
| Fitting `α`, panel composition → `NSL.pdf`            | [scaling_law/plot.ipynb](scaling_law/plot.ipynb) (also saves `NSL_dataset.pdf`, `NSL_width.pdf`)                                                                                                                            |

### Fig. 6 — Runtime benchmark of the hybrid algorithm (`runtime_plot.pdf`, Appendix `app:algorithm`)

Uniformly-sampled `(d, m=5)` cubes, `k=5`, measuring wall-clock runtime.

| Step         | File                                                                                                                                                                                               |
| ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Timing sweep | [runtime_benchmark/runtime.py](runtime_benchmark/runtime.py) + [runtime_benchmark/data_gen.py](runtime_benchmark/data_gen.py) → [runtime_benchmark/time_list.csv](runtime_benchmark/time_list.csv) |
| Plot         | [runtime_benchmark/runtime_plot.ipynb](runtime_benchmark/runtime_plot.ipynb) → `runtime_plot.pdf`                                                                                                  |

### Fig. 7 — Dynamical LTH for multi-head attention (`attention.pdf`, Appendix `app:attention`)

In-context learning of piecewise-linear 1-D functions; `d_heads = 4000` head MHA compared with compressed `d' = 800` head MHA and a random-head-subset baseline.

| Step                                                    | File                                                                                                                                                        |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| MHA module + `compress_mha`                             | [transformer/mha.py](transformer/mha.py)                                                                                                                    |
| Piecewise-linear ICL task (used for the final figure)   | [transformer/ICL_piecewise.py](transformer/ICL_piecewise.py) (dataset: `generate_piecewise_linear_params`, `eval_piecewise_linear`, `sample_episode_batch`) |
| Alternative smooth-function ICL variant (earlier draft) | [transformer/ICL.py](transformer/ICL.py), [transformer/compress_dynamics.py](transformer/compress_dynamics.py)                                              |
| Teacher and helper utilities                            | [transformer/teacher.py](transformer/teacher.py), [transformer/common.py](transformer/common.py)                                                            |
| Plot → `figures/attention.pdf`                          | [transformer/plot.ipynb](transformer/plot.ipynb) consuming `transformer/icl_piecewise_losses_*.csv`                                                         |

### Fig. 8 — polylog-compression error scaling (`error_scaling_polylog.pdf`, Appendix)

For each `d`, sweeps `k` up to `dstop/2 ≈ 60 log d` to find the smallest error; `dstop(d) = 120 log d`. Demonstrates that the polylog-rate predicted by Thm. `errorbound_polylog` can be attained numerically in `m=1,2`.

| Step              | File                                                                                                                                         |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| Compression sweep | [error_scaling/polylog_error.py](error_scaling/polylog_error.py) → `polylog_error_1d.csv`, `polylog_error_2d.csv`                            |
| Plot              | [error_scaling/plot.ipynb](error_scaling/plot.ipynb) → `polylog_m1.pdf`, `polylog_m2.pdf`, combined into `figures/error_scaling_polylog.pdf` |

### Shared core

All figures (except Fig. 1) depend on:

- [compressor.py](compressor.py) — the `Compressor` class implementing `_find_best_subset` (FAISS greedy), `_reduce_compute` (Carathéodory peeling via `find_null_vec`), and the k-means/greedy hybrid `compress(k, dstop=…)`.
- FAISS CPU for nearest-neighbour clustering; NumPy / SciPy for null-vector computation; scikit-learn `MiniBatchKMeans`; joblib for parallel reduction.
- PyTorch for all training experiments (Figs. 3–5, 7).

### Stale / superseded directories (not used by final figures)

- [supplementary/](supplementary/) — a trimmed copy of the core scripts packaged for anonymous submission; duplicates [new/](new/).
