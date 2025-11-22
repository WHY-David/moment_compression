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
    return y


# determine the final data set size
def dstop(d):
    # return int(120*np.log(d))
    return int(20*(np.log(d))**2)

def run_trial(args):
    m, d, t = args
    seed_data = 0
    rng = np.random.default_rng(seed_data + m + d + t)
    x = 2*rng.random((m, 10)) - 1
    data = generate_random_data(d, m, rng)
    f_orig = f(data, x)
    least_error = None
    
    for k in range(1, int(np.sqrt(2*dstop(d)))-1):
        worker = Compressor(data)
        c, W = worker.compress(k, dstop = dstop(d), print_progress=False)
        f_comp = f(W, x, weights=c)
        new_error = abs(f_comp - f_orig)
        if least_error is None:
            least_error = new_error
        else:
            if new_error < least_error:
                least_error = new_error
            else:
                print(f"Completed m = {m}, k = {k}, d = {d}, t = {t}")
                return m, k, d, t, least_error




if __name__ == "__main__":
    # parameters
    mlist = [2,]
    d_list = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
    trials_per_d = 5
    # num_samples = 10

    tasks = [(m, d, t) for m in mlist for d in d_list for t in range(trials_per_d)]

    out_path = "polylog_error_list.csv"
    file_exists = os.path.exists(out_path)

    # Open once and append rows as each trial completes; flush+fsync for durability
    with open(out_path, "a", newline="") as outfile:
        writer = csv.writer(outfile)
        if not file_exists:
            writer.writerow(["m", "k", "d", "t", "least_error"])  # header
            outfile.flush()
            os.fsync(outfile.fileno())

        for args in tasks:
            try:
                m, k, d, t, least_error = run_trial(args)
                writer.writerow([m, k, d, t, least_error])
                # Ensure the write hits disk so prior results persist even on error
                outfile.flush()
                os.fsync(outfile.fileno())
            except Exception as e:
                # Log the error but continue with subsequent tasks
                print(f"Error on args={args}: {e}", flush=True)
