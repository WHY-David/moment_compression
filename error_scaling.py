import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from new.compressor import Compressor

import os
import csv

def f(data: np.ndarray, x:np.ndarray, weights=None) -> float:
    d, m = data.shape
    assert x.shape[0] == m
    y = data @ x
    y = 1./(1+np.exp(-1*y))
    # y = np.sin(2*np.pi*y)
    y = np.mean(y, axis=1)
    if weights is not None:
        assert weights.size == d
        y = y @ weights / sum(weights)
    else:
        y = np.mean(y)
    # return np.sin(2*np.pi*y)
    return y


# determine the final data set size
def dstop(d):
    # return int(3.5*d**0.5)
    return int(0.1*d)
    # return 35

def run_trial(args):
    m, k, d, t = args
    seed_data = 0
    rng = np.random.default_rng(seed_data + m + d + t)
    x = 2*rng.random((m, 10)) - 1
    data = 2*rng.random((d, m)) - 1
    orig = f(data, x)
    
    worker = Compressor(data)
    c, W = worker.compress(k, dstop = dstop(d), print_progress=False)
    comp = f(W, x, weights=c)
    print(f"Completed m = {m}, k = {k}, d = {d}, t = {t}")
    return m, k, d, orig, comp




if __name__ == "__main__":
    # parameters
    # m = 4   # dimension of each point
    mlist = [1, 2, 3, 4, 5]
    d_list = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
    k_list = [1, 2, 3, 4, 5, 6]
    trials_per_d = 20
    num_samples = 10

    tasks = [(m, k, d, t) for m in mlist for k in k_list for d in d_list for t in range(trials_per_d)]

    out_path = "error_list.csv"
    file_exists = os.path.exists(out_path)

    # Open once and append rows as each trial completes; flush+fsync for durability
    with open(out_path, "a", newline="") as outfile:
        writer = csv.writer(outfile)
        if not file_exists:
            writer.writerow(["m", "k", "d", "orig", "comp"])  # header
            outfile.flush()
            os.fsync(outfile.fileno())

        for args in tasks:
            try:
                m, k, d, orig, comp = run_trial(args)
                writer.writerow([m, k, d, orig, comp])
                # Ensure the write hits disk so prior results persist even on error
                outfile.flush()
                os.fsync(outfile.fileno())
            except Exception as e:
                # Log the error but continue with subsequent tasks
                print(f"Error on args={args}: {e}", flush=True)

    # # compute max errors for each k
    # max_errors = {
    #     k: [np.max(results[k][d]) for d in d_list]
    #     for k in k_list
    # }

    # filename = f'figures/linear_m{m}.pdf'

    # # plot max error vs. d for each k with fitted power-law exponents
    # axes_width_pt = 360
    # axes_height_pt = 0.7 * axes_width_pt
    # left_margin_pt, right_margin_pt = 35, 35
    # bottom_margin_pt, top_margin_pt = 30, 20
    # fig_width_pt = left_margin_pt + axes_width_pt + right_margin_pt
    # fig_height_pt = bottom_margin_pt + axes_height_pt + top_margin_pt

    # fig = plt.figure(figsize=(fig_width_pt/72, fig_height_pt/72))
    # ax = fig.add_axes([
    #     left_margin_pt/fig_width_pt,
    #     bottom_margin_pt/fig_height_pt,
    #     axes_width_pt/fig_width_pt,
    #     axes_height_pt/fig_height_pt
    # ])

    # d_vals = np.array(d_list)
    # for k in k_list:
    #     errs = max_errors[k]
    #     # fit power-law exponent α_k: err ∝ d^α_k
    #     coeffs = np.polyfit(np.log(d_vals), np.log(errs), 1)
    #     alpha_k = coeffs[0]
    #     plt.plot(d_list, errs, marker='o', linestyle='-', label=f"k={k}, power={alpha_k:.3f}")

    # plt.xscale('log')
    # plt.yscale('log')
    # plt.xlabel(r"Data set size $d$")
    # plt.ylabel(r"$\max|f(\theta')-f(\theta)|$")
    # plt.title(r"Compression: $d \to 0.1d$. "+f"Data dimension m={m}")
    # plt.legend()
    # plt.tight_layout()
    # # plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0)
    # plt.show()