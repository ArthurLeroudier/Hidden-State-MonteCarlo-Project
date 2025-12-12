import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.load_data import load_wta
from models.kalmanfilters import ExtendedKalmanFilters

match_times, match_player_indices, _, players_id_to_name_dict, _ = load_wta()

def f(tau, sigma0):

    wta_kalman = ExtendedKalmanFilters(match_times,
                                       match_player_indices,
                                       players_id_to_name_dict=players_id_to_name_dict,
                                       tau=tau,
                                       sigma0=sigma0)
    return wta_kalman.compute_llh(modeltype="ScalarVariance")

tau_axis = np.logspace(-3, -1, 30)
sigma0_axis = np.logspace(-2, 0, 30)

X, Y = np.meshgrid(tau_axis, sigma0_axis)
Z = np.zeros_like(X)

save_path = "experiments/graphs/heatmap_data.npz"

if os.path.exists(save_path):
    data = np.load(save_path)
    Z = data["Z"]
    done_mask = ~np.isnan(Z)
else:
    Z[:] = np.nan
    done_mask = np.zeros_like(Z, dtype=bool)

total = Z.size
remaining = np.sum(~done_mask)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        if done_mask[i, j]:
            continue

        Z[i, j] = f(X[i, j], Y[i, j])
        print("llh: ", Z[i, j], " for tau: ", X[i, j], " sigma0: ", Y[i, j])
        done_mask[i, j] = True

        if (i * X.shape[1] + j) % 20 == 0:
            np.savez(save_path, Z=Z, tau_axis=tau_axis, sigma0_axis=sigma0_axis)

np.savez(save_path, Z=Z, tau_axis=tau_axis, sigma0_axis=sigma0_axis)

plt.figure(figsize=(8, 6))
pcm = plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\tau$ (log scale)")
plt.ylabel(r"$\sigma_0$ (log scale)")
plt.title("Log-likelihood of WTA data under Extended Kalman Filter model")
plt.tight_layout()
plt.savefig("experiments/graphs/wta_ekf_loglikelihood_heatmap.png")

