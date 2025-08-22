import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import csv

from data_gen import generate_points_lattice, generate_random_data

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from compressor import Compressor

import argparse



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
    data = generate_random_data(d, m, rng)
    orig = f(data, x)
    
    worker = Compressor(data)
    c, W = worker.compress(k, dstop = dstop(d), print_progress=False)
    comp = f(W, x, weights=c)
    print(f"Completed m = {m}, k = {k}, d = {d}, t = {t}")
    return m, k, d, orig, comp




if __name__ == "__main__":
    # CLI for running either all tasks (default) or a single task selected by index or explicit params.
    parser = argparse.ArgumentParser(description="Run error-scaling tasks; supports Slurm job arrays.")
    parser.add_argument("--task-index", type=int, default=None,
                        help="0-based index into the cartesian product of (m,k,d,trial). If provided, runs exactly that task.")
    parser.add_argument("--m", type=int, default=None,
                        help="If provided together with --k, --d, --trial, run exactly this single task.")
    parser.add_argument("--k", type=int, default=None,
                        help="Hidden size parameter for the single-task mode.")
    parser.add_argument("--d", type=int, default=None,
                        help="Dataset size for the single-task mode.")
    parser.add_argument("--trial", type=int, default=None,
                        help="Trial index t for the single-task mode.")
    parser.add_argument("--out", dest="out_path", default="error_uniform.csv",
                        help="Output CSV path (default: error_uniform.csv).")

    args = parser.parse_args()

    # parameters (default grid)
    # mlist = [1, 2, 3, 4, 5]
    # d_list = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
    # k_list = [1, 2, 3, 4, 5, 6]
    mlist = [5]
    d_list = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
    k_list = [4, 5, 6]
    trials_per_d = 20
    num_samples = 10  # kept for compatibility; unused in current pipeline

    # Build the full task list in a deterministic order
    full_tasks = [(m, k, d, t) for m in mlist for k in k_list for d in d_list for t in range(trials_per_d)]

    # Decide which task(s) to run
    selected_tasks = None

    # Mode 1: explicit single-task via --m/--k/--d/--trial
    if any(v is not None for v in (args.m, args.k, args.d, args.trial)):
        if None in (args.m, args.k, args.d, args.trial):
            raise SystemExit("If using --m/--k/--d/--trial, you must provide all four.")
        selected_tasks = [(args.m, args.k, args.d, args.trial)]
    else:
        # Mode 2: index-based single-task via --task-index, defaulting to SLURM_ARRAY_TASK_ID if present
        index = args.task_index
        if index is None:
            env_id = os.environ.get("SLURM_ARRAY_TASK_ID")
            if env_id is not None:
                try:
                    index = int(env_id)
                except ValueError:
                    index = None
        if index is not None:
            if index < 0 or index >= len(full_tasks):
                raise SystemExit(f"--task-index {index} out of range [0, {len(full_tasks)-1}]")
            selected_tasks = [full_tasks[index]]
        else:
            # Mode 3: run all tasks (original behavior)
            selected_tasks = full_tasks

    out_path = args.out_path
    file_exists = os.path.exists(out_path)

    # Open once and append rows as each task completes; flush+fsync for durability
    with open(out_path, "a", newline="") as outfile:
        writer = csv.writer(outfile)
        if not file_exists:
            writer.writerow(["m", "k", "d", "orig", "comp"])  # header
            outfile.flush()
            os.fsync(outfile.fileno())

        for task in selected_tasks:
            try:
                m, k, d, orig, comp = run_trial(task)
                writer.writerow([m, k, d, orig, comp])
                # Ensure the write hits disk so prior results persist even on error
                outfile.flush()
                os.fsync(outfile.fileno())
            except Exception as e:
                # Log the error but continue with subsequent tasks
                print(f"Error on task={task}: {e}", flush=True)
