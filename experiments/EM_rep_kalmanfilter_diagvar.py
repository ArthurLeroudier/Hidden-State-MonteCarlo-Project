import numpy as np
import matplotlib.pyplot as plt
from data.load_data import load_wta
from models.kalmanfilters import ExtendedKalmanFilters
from models.EM_kalmanfilters import EM_KalmanFilter
import os

match_times, match_player_indices, _, players_id_to_name_dict, _ = load_wta()

tau_init = 0.01
sigma0_init = 0.1
n_iter = 1000

wta_kalman = ExtendedKalmanFilters(
    match_times,
    match_player_indices,
    players_id_to_name_dict=players_id_to_name_dict,
    tau=tau_init,
    sigma0=sigma0_init
)

sigma0_traj, tau_traj = EM_KalmanFilter(wta_kalman, n_iter=n_iter)

save_dir = "experiments/graphs"
os.makedirs(save_dir, exist_ok=True)

np.save(os.path.join(save_dir, "tau_traj.npy"), tau_traj)
np.save(os.path.join(save_dir, "sigma0_traj.npy"), sigma0_traj)

tau_axis = np.logspace(-3, -1, 30)
sigma0_axis = np.logspace(-2, 0, 30)

plt.figure(figsize=(8, 6))
plt.plot(tau_traj, sigma0_traj, marker='o', color='red', label='EM trajectory')
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\tau$")
plt.ylabel(r"$\sigma_0$")
plt.xlim(tau_axis[0], tau_axis[-1])
plt.ylim(sigma0_axis[0], sigma0_axis[-1])
plt.title("Trajectory of EM for WTA EKF DiagonalVariance")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "em_trajectory_diagvar.png"))
plt.show()

