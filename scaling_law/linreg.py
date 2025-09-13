import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='Helvetica', size=8)
import csv

from common import fix_random_seed, make_canvas
from data_gen import generate_data

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from compressor import Compressor



def fit_eval(data:np.ndarray, truth, weights=None):
    d, m = data.shape
    X = data[:, :-1]
    z = data[:, -1]

    if weights is not None:
        assert len(weights) == d
        wsqrt = np.sqrt(weights)
        X *= wsqrt[:, None]
        z *= wsqrt

    fit, _, _, _ = np.linalg.lstsq(X, z, rcond=None)
    return 4/3 * sum((fit-truth)**2)




if __name__ == "__main__":
    seed = 42
    rep = 5
    fix_random_seed(seed)

    # train size
    dlist = [2**x for x in range(7, 17)]
    
    # compression rate
    dstoplist = [int(d**(1/3)) for d in dlist]
    k = 2

    train_noise = 10.

    truth = np.array([1., -2.]) # z = ax+by+noise
    f = lambda x, y: truth[0]*x + truth[1]*y

    losses = []
    losses_cp = []

    for n, d in enumerate(dlist):
        loss_temp = np.zeros(rep)
        loss_cp_temp = np.zeros(rep)
        for trial in range(rep):
            data = generate_data(d, f=f, noise=train_noise, seed=seed+trial+n, return_tensor=False)
            cp = Compressor(data, random_state=d+trial)
            c_, data_cp = cp.compress(k, dstop=dstoplist[n])
            loss_temp[trial] = fit_eval(data, truth)
            loss_cp_temp[trial] = fit_eval(data_cp, truth, weights=c_)
        loss = np.mean(loss_temp)
        loss_cp = np.mean(loss_cp_temp)
        losses.append(loss)
        losses_cp.append(loss_cp)
        print(f"Completed d = {d}")

    # plots
    fig, axs = make_canvas(rows=1, cols=1, axes_width_pt=300)

    # Plot Train Loss vs. epoch
    axs.plot(dlist, losses, marker='^', label='Original')
    axs.plot(dstoplist, losses_cp, marker='o', label='Compressed')
    axs.set_xscale('log')
    axs.set_yscale('log')
    axs.legend()
    plt.tight_layout()

    # filename = f'CPTDS/linear_{algo_name}_d{d}_dstop{dstop}_k{k}_noise{train_noise}_hidden{hidden_dim}_bs{batch_size}_lr{lr}'
    # with open(filename + '.csv', 'w', newline='') as csvfile:
    #     writer = csv.writer(csvfile)
    #     # Write header
    #     writer.writerow([
    #         'epoch',
    #         'train_loss_orig',
    #         'test_loss_orig',
    #         'train_loss_cp',
    #         'test_loss_cp',
    #         'train_loss_naive',
    #         'test_loss_naive'
    #     ])
    #     # Write data rows
    #     for i in range(len(epoch_range)):
    #         writer.writerow([
    #             epoch_range[i],
    #             train_loss_orig[i],
    #             test_loss_orig[i],
    #             train_loss_cp[i],
    #             test_loss_cp[i],
    #             train_loss_naive[i],
    #             test_loss_naive[i]
    #         ])
    # plt.savefig(filename+'.pdf', format='pdf', bbox_inches='tight', pad_inches=0)
    plt.show()
