import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.load_data import load_wta, load_wta_sub
from models.kalmanfilters import ExtendedKalmanFilters

match_times, match_player_indices, _, players_id_to_name_dict, _ = load_wta_sub(nb_players=100)
modelType = "DiagonalVariance"

print(len(match_player_indices), " matches loaded for WTA subset with 100 players.")

plot_definition = 50

def f(tau, sigma0):
    wta_kalman = ExtendedKalmanFilters(
        match_times,
        match_player_indices,
        players_id_to_name_dict=players_id_to_name_dict,
        tau=tau,
        sigma0=sigma0
    )
    wta_kalman.filtering(modelType=modelType)
    return wta_kalman.log_likelihood

tau_axis = np.logspace(-6, 3, plot_definition)
sigma0_axis = np.logspace(-6, 0, plot_definition)

X, Y = np.meshgrid(tau_axis, sigma0_axis)
Z = np.zeros_like(X)

for i in tqdm(range(X.shape[0])):
    for j in range(X.shape[1]):
        Z[i, j] = float(f(X[i, j], Y[i, j]))

save_dir = "experiments/graphs"
os.makedirs(save_dir, exist_ok=True)

np.savez(os.path.join(save_dir, f"wta_ekf_loglikelihood_{modelType}.npz"),
         tau_axis=tau_axis, sigma0_axis=sigma0_axis, loglikelihood=Z)

plt.figure(figsize=(8, 6))
pcm = plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')

plt.colorbar(pcm, label="Log-vraisemblance")

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\tau$ (log scale)")
plt.ylabel(r"$\sigma_0$ (log scale)")
plt.title("Log-likelihood of WTA data under Extended Kalman Filter model")
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f"wta_ekf_loglikelihood_{modelType}.png"))
plt.show()