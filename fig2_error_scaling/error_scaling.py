import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import csv

from data_gen import generate_points_lattice, generate_random_data

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compressor import Compressor



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
    return int(d**0.5)
    # return int(0.1*d)
    # return 35

def run_trial(args):
    m, k, d, t = args
    seed_data = 0
    rng = np.random.default_rng(seed_data + m + d + t)
    x = 2*rng.random((m, 10)) - 1
    data = generate_random_data(d, m, rng)
    orig = f(data, x)
    
    worker = Compressor(data)
    c, W = worker.compress(k, dstop = dstop(d), print_progress=False)
    comp = f(W, x, weights=c)
    print(f"Completed m = {m}, k = {k}, d = {d}, t = {t}")
    return m, k, d, orig, comp




if __name__ == "__main__":
    # parameters
    # mlist = [1, 2, 3, 4, 5]
    # d_list = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
    # k_list = [1, 2, 3, 4, 5, 6]
    mlist = [3]
    d_list = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
    k_list = [4, 5, 6]
    trials_per_d = 10
    num_samples = 10

    tasks = [(m, k, d, t) for m in mlist for k in k_list for d in d_list for t in range(trials_per_d)]

    out_path = "sqrt_error_list.csv"
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
