import jax.numpy as jnp
from tqdm import tqdm

from models.kalmanfilters import ExtendedKalmanFilters

def EM_KalmanFilter(
    filter_obj,
    modelType="DiagonalVariance",
    n_iter=1000,
    tol_sigma0=1e-4,
    tol_tau=1e-5
    ):

    N = filter_obj.N
    K = filter_obj.K

    sigma0 = filter_obj.sigma0
    tau = filter_obj.tau

    sigma0_traj = []
    tau_traj = []

    dt = filter_obj.matches_info[1:, 2].reshape(1, -1)

    for _ in tqdm(range(n_iter)):
        sigma0_old = sigma0
        tau_old = tau

        filter_obj.sigma0 = sigma0
        filter_obj.tau = tau
        filter_obj.updated = False
        filter_obj.filtering(modelType=modelType)
        filter_obj.smoothing(modelType=modelType)

        mu_s = filter_obj.mu_smooth
        v_s = filter_obj.v_smooth
        mu_cross = filter_obj.mu_cross_smooth

        sigma0 = jnp.mean(v_s[:, 0] + mu_s[:, 0] ** 2)

        mu_prev = mu_s[:, :-1]
        mu_next = mu_s[:, 1:]
        v_prev = v_s[:, :-1]
        v_next = v_s[:, 1:]

        numerator = v_prev + mu_prev ** 2 - 2.0 * mu_cross + v_next + mu_next ** 2
        tau_sq_matrix = jnp.where(dt != 0, numerator / dt, 0.0)
        tau = jnp.sqrt(jnp.sum(tau_sq_matrix) / (N * K))

        sigma0_traj.append(float(sigma0))
        tau_traj.append(float(tau))

        if jnp.abs(sigma0 - sigma0_old) < tol_sigma0 and jnp.abs(tau - tau_old) < tol_tau:
            print("Converged")
            break

    return sigma0_traj, tau_traj


