import numpy as np

def generate_random_data(d, m, rng):
    return 2*rng.random((d, m)) - 1

# ---------- utility ----------
def pairwise_min_dist(X: np.ndarray) -> float:
    """Return min pairwise Euclidean distance among rows of X."""
    diff = X[:, None, :] - X[None, :, :]
    r2 = np.sum(diff**2, axis=-1)
    np.fill_diagonal(r2, np.inf)
    return float(np.sqrt(np.min(r2)))

def generate_points_lattice(d: int, m: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build a k-per-dimension lattice in [-1,1]^m with k = ceil(d**(1/m)),
    then uniformly remove points at random until exactly d remain.

    Returns: (d, m) array.
    """
    k = int(np.ceil(d ** (1.0 / m)))
    grid_1d = np.linspace(-1.0, 1.0, k)
    mesh = np.meshgrid(*([grid_1d] * m), indexing="ij")
    lattice = np.stack([g.reshape(-1) for g in mesh], axis=1)  # (k^m, m)
    N = lattice.shape[0]

    if N == d:
        return lattice
    # Randomly keep exactly d points (equivalent to removing N-d)
    keep = rng.choice(N, size=d, replace=False)
    return lattice[keep, :]


# ---------- example usage ----------
if __name__ == "__main__":
    d, m = 100000, 3
    rng = np.random.default_rng(0)

    P_lattice = generate_points_lattice(d, m, rng)
    # P_repulse = generate_points_repulsion(d, m, rng)

    print("Lattice+FPS min dist:", "8")


    # import matplotlib.pyplot as plt
    # plt.scatter(P_lattice[:, 0], P_lattice[:, 1], s=1, alpha=0.5)
    # plt.show()