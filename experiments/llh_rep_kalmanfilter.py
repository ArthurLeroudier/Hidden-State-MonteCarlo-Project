import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.load_data import load_wta
from models.kalmanfilters import ExtendedKalmanFilters

match_times, match_player_indices, _, players_id_to_name_dict, _ = load_wta()
modelType = "DiagonalVariance"

plot_definition = 50

def f(tau, sigma0):

    wta_kalman = ExtendedKalmanFilters(match_times,
                                       match_player_indices,
                                       players_id_to_name_dict=players_id_to_name_dict,
                                       tau=tau,
                                       sigma0=sigma0)
    
    wta_kalman.filtering(modelType=modelType)

    return wta_kalman.log_likelihood

tau_axis = np.logspace(-3, 0, 50)
sigma0_axis = np.logspace(-2, 0, 50)

X, Y = np.meshgrid(tau_axis, sigma0_axis)
Z = np.zeros_like(X)


for i in tqdm(range(X.shape[0])):
    for j in range(X.shape[1]):

        Z[i, j] = float(f(X[i, j], Y[i, j]))

plt.figure(figsize=(8, 6))
pcm = plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\tau$ (log scale)")
plt.ylabel(r"$\sigma_0$ (log scale)")
plt.title("Log-likelihood of WTA data under Extended Kalman Filter model")
plt.tight_layout()
plt.savefig(f"experiments/graphs/wta_ekf_loglikelihood_{modelType}.png")

