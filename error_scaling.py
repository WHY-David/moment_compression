import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
# from moment_matching import compress, compress_naive
from new.compressor import Compressor


def f(data: np.ndarray, x:np.ndarray, weights=None) -> float:
    d, m = data.shape
    assert x.shape[0] == m
    y = data @ x
    y = np.exp(y)
    y = np.mean(y, axis=1)
    if weights is not None:
        assert weights.size == d
        y = y @ weights / sum(weights)
    else:
        y = np.mean(y)
    return np.sin(2*np.pi*y)



# parameters
m = 3   # dimension of each point
d_list = [1000, 2000, 4000, 8000, 16000, 32000]
trials_per_d = 5
num_samples = 10
seed_data = 0
seed_f = 42
k_list = [1, 2, 3, 4]
filename = f'figures/linear_m{m}.pdf'


# initialize results dict: results[k][d] = list of errors
results = {k: {d: [] for d in d_list} for k in k_list}


# determine the final data set size
def dstop(d):
    # return int(3.5*d**0.5)
    return int(0.35*d)
    # return 35

def run_trial(args):
    k, d, t = args
    rng = np.random.default_rng(seed_data + t)
    x = 2*rng.random((m, 10)) - 1
    data = 2*rng.random((d, m)) - 1
    orig = f(data, x)
    
    worker = Compressor(data)
    c, W = worker.compress(k, dstop = dstop(d), print_progress=True)
    comp = f(W, x, weights=c)
    return k, d, abs(comp - orig)




if __name__ == "__main__":
    tasks = [(k, d, t) for k in k_list for d in d_list for t in range(trials_per_d)]
    with mp.Pool() as pool:
        results_list = pool.map(run_trial, tasks)

    # populate results dictionary
    for k, d, err in results_list:
        results[k][d].append(err)

    # compute max errors for each k
    max_errors = {
        k: [np.max(results[k][d]) for d in d_list]
        for k in k_list
    }

    # plot max error vs. d for each k with fitted power-law exponents
    axes_width_pt = 360
    axes_height_pt = 0.7 * axes_width_pt
    left_margin_pt, right_margin_pt = 35, 35
    bottom_margin_pt, top_margin_pt = 30, 20
    fig_width_pt = left_margin_pt + axes_width_pt + right_margin_pt
    fig_height_pt = bottom_margin_pt + axes_height_pt + top_margin_pt

    fig = plt.figure(figsize=(fig_width_pt/72, fig_height_pt/72))
    ax = fig.add_axes([
        left_margin_pt/fig_width_pt,
        bottom_margin_pt/fig_height_pt,
        axes_width_pt/fig_width_pt,
        axes_height_pt/fig_height_pt
    ])

    d_vals = np.array(d_list)
    for k in k_list:
        errs = max_errors[k]
        # fit power-law exponent α_k: err ∝ d^α_k
        coeffs = np.polyfit(np.log(d_vals), np.log(errs), 1)
        alpha_k = coeffs[0]
        plt.plot(d_list, errs, marker='o', linestyle='-', label=f"k={k}, power={alpha_k:.3f}")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"Data set size $d$")
    plt.ylabel(r"$\max|f(\theta')-f(\theta)|$")
    plt.title(r"Compression: $d \to 0.35d$. "+f"Data dimension m={m}")
    plt.legend()
    plt.tight_layout()
    # plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()