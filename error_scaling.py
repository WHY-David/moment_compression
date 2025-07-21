import numpy as np
from moment_matching import compress, compress_naive

def f(data: np.ndarray, num_samples: int = 100, seed: int = 0) -> float:
    """
    Given data of shape (d, m) representing {w_i}_{i=1}^d ⊂ ℝ^m,
    returns the average over num_samples of
        ∑_{i=1}^d sigmoid(w_i · x),
    with x drawn i.i.d. ~ N(0,1)^m using the fixed seed.
    """
    # data.shape = (d, m)
    d, m = data.shape

    # sigmoid function
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

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
    # sigmoid(2*D), weighted sum
    S = (1.0 / (1.0 + np.exp(-D))) @ c  # shape (num_samples,)
    return S.mean()


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # parameters
    m = 2   # dimension of each point
    d_list = [100, 200, 400, 800, 1600, 3200]
    trials_per_d = 5
    num_samples = 10
    seed_data = 0
    seed_f = 42
    k_list = [1, 2, 3, 4]

    # initialize results dict: results[k][d] = list of errors
    results = {k: {d: [] for d in d_list} for k in k_list}

    for k in k_list:
        for d in d_list:
            for t in range(trials_per_d):
                rng = np.random.default_rng(seed_data + t)
                data = rng.random((d, m))
                # compute original f
                orig = f(data, num_samples=num_samples, seed=seed_f)
                # compress_naive to degree k
                c, W = compress(data, k)
                # compute weighted f on compressed support
                comp = f_weighted(c, W, num_samples=num_samples, seed=seed_f)
                # record absolute error
                results[k][d].append(abs(comp - orig))

    # compute mean errors for each k
    mean_errors = {
        k: [np.mean(results[k][d]) for d in d_list]
        for k in k_list
    }

    # plot average error vs. d for each k with fitted power-law exponents
    plt.figure(figsize=(8, 5))
    d_vals = np.array(d_list)
    for k in k_list:
        errs = mean_errors[k]
        # fit power-law exponent α_k: err ∝ d^α_k
        coeffs = np.polyfit(np.log(d_vals), np.log(errs), 1)
        alpha_k = coeffs[0]
        plt.plot(d_list, errs, marker='o', label=f"k={k}, power={alpha_k:.3f}")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r"Data set size $d$")
    plt.ylabel(r"$|f_{\mathrm{comp}} - f_{\mathrm{orig}}|$")
    plt.title(r"Error vs. $d$ for matching degrees 1-4")
    plt.legend()
    plt.tight_layout()
    plt.show()