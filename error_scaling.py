import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from moment_matching import compress, compress_naive

# sigmoid function
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def f(data: np.ndarray, num_samples: int = 100, seed: int = 0) -> float:
    """
    Given data of shape (d, m) representing {w_i}_{i=1}^d ⊂ ℝ^m,
    returns the average over num_samples of
        ∑_{i=1}^d sigmoid(w_i · x),
    with x drawn i.i.d. ~ N(0,1)^m using the fixed seed.
    """
    # data.shape = (d, m)
    d, m = data.shape

    # draw all x's at once for efficiency
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((num_samples, m))    # shape (num_samples, m)

    # compute dot products: shape (num_samples, d)
    D = X @ data.T

    # apply sigmoid and sum over i for each sample
    sums = sigmoid(D).sum(axis=1)                 # shape (num_samples,)

    # return the average over samples
    return sums.mean()


# define a weighted version of f to handle compress outputs
def f_weighted(c, W, num_samples: int = 100, seed: int = 0) -> float:
    """
    c : weights array of length L
    W : array shape (L, m)
    returns average over num_samples of sum_j c_j * sigmoid(2 * W_j · x)
    """
    rng = np.random.default_rng(seed)
    m = W.shape[1]
    X = rng.standard_normal((num_samples, m))
    # compute dot products: shape (num_samples, L)
    D = X @ W.T

    S = sigmoid(D) @ c  # shape (num_samples,)
    return S.mean()





# parameters
m = 2   # dimension of each point
d_list = [100, 200, 400, 800, 1600, 3200, 6400]
trials_per_d = 10
num_samples = 10
seed_data = 0
seed_f = 42
k_list = [1, 2, 3, 4]
plot_error_bars = True
filename = 'figures/sqrt_m2.pdf'


# initialize results dict: results[k][d] = list of errors
results = {k: {d: [] for d in d_list} for k in k_list}


# determine the final data set size
def dstop(d):
    return int(3.5*d**0.5)
    # return int(0.35*d)
    # return 35

def run_trial(args):
    k, d, t = args
    rng = np.random.default_rng(seed_data + t)
    data = rng.random((d, m))
    orig = f(data, num_samples=num_samples, seed=seed_f)
    c, W = compress(data, k, dstop = dstop(d), index_type='flat')
    # c, W = compress_naive(data, k, dstop=dstop(d))
    comp = f_weighted(c, W, num_samples=num_samples, seed=seed_f)
    return k, d, abs(comp - orig)




if __name__ == "__main__":
    tasks = [(k, d, t) for k in k_list for d in d_list for t in range(trials_per_d)]
    with mp.Pool() as pool:
        results_list = pool.map(run_trial, tasks)

    # populate results dictionary
    for k, d, err in results_list:
        results[k][d].append(err)

    # compute mean errors for each k
    mean_errors = {
        k: [np.mean(results[k][d]) for d in d_list]
        for k in k_list
    }

    # compute standard deviations for each k and d
    std_errors = {
        k: [np.std(results[k][d]) for d in d_list]
        for k in k_list
    }

    # plot average error vs. d for each k with fitted power-law exponents
    # fixed axes frame size in points (same as plot_both)
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
        errs = mean_errors[k]
        stds = std_errors[k]
        # fit power-law exponent α_k: err ∝ d^α_k
        coeffs = np.polyfit(np.log(d_vals), np.log(errs), 1)
        alpha_k = coeffs[0]
        if plot_error_bars:
            plt.errorbar(
                d_list, errs, yerr=stds, marker='o',
                linestyle='-', capsize=3,
                label=f"k={k}, power={alpha_k:.3f}"
            )
        else:
            plt.plot(d_list, errs, marker='o', linestyle='-', label=f"k={k}, power={alpha_k:.3f}")

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"Data set size $d$")
    plt.ylabel(r"$|f_{\mathrm{comp}} - f_{\mathrm{orig}}|$")
    plt.title(r"Compression: $d \to 3.5\sqrt{d}$. "+f"Data dimension m={m}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()