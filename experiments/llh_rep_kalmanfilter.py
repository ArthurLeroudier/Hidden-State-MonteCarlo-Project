import matplotlib.pyplot as plt
import numpy as np

from data.load_data import load_wta
from models.kalmanfilters import ExtendedKalmanFilters

def f(tau, sigma0):
    match_times, match_player_indices, _, players_id_to_name_dict, _ = load_wta()
    wta_kalman = ExtendedKalmanFilters(match_times,
                                    match_player_indices,
                                    players_id_to_name_dict=players_id_to_name_dict,
                                    tau=tau,
                                    sigma0 = sigma0,
                                    )
    
    loglikelihood = wta_kalman.compute_llh(modeltype="FixedVariance")
    return loglikelihood


tau_axis = np.logspace(-3, -1, 30, base=10)
sigma0_axis = np.logspace(-2, 0, 30, base=10)

X, Y = np.meshgrid(tau_axis, sigma0_axis)

Z = np.zeros_like(X)

for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i,j] = f(X[i,j], Y[i,j])


plt.figure(figsize=(8, 6))


pcm = plt.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')

plt.xscale('log')
plt.yscale('log')

plt.xlabel(r"$\tau$ (log scale)")
plt.ylabel(r"$\sigma_0$ (log scale)")
plt.title("Log-likelihood of WTA data under Extended Kalman Filter model")

plt.colorbar(pcm, label="Log-likelihood value")
plt.tight_layout()
plt.show()
