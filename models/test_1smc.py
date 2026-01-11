import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.load_data_numpy import load_wta_numpy
from models.pf_2 import PFParams, run_filter, run_em_sigma0_tau


def compute_grid_smc(
    match_times,
    match_player_indices,
    n_players,
    tau_axis,
    sigma0_axis,
    s_fixed=1.0,
    n_particles_grid=300,
    seed=0,
    cache_path="fig3_smc_grid_cache.npz",
):
    """
    Compute Z[i,j] = loglik(tau_axis[j], sigma0_axis[i]) with PF.
    Cached to disk to avoid recomputation.
    """
    # Try cache
    try:
        data = np.load(cache_path)
        Z = data["Z"]
        if Z.shape == (len(sigma0_axis), len(tau_axis)):
            return Z
    except Exception:
        pass

    Z = np.empty((len(sigma0_axis), len(tau_axis)), dtype=float)

    for i in tqdm(range(len(sigma0_axis)), desc="Grid rows (sigma0)"):
        sigma0 = float(sigma0_axis[i])
        for j in range(len(tau_axis)):
            tau = float(tau_axis[j])
            params = PFParams(sigma0=sigma0, tau=tau, s=s_fixed)

            _, loglik, _ = run_filter(
                match_times=match_times,
                match_player_indices=match_player_indices,
                n_players=n_players,
                params=params,
                n_particles=n_particles_grid,
                seed=seed,
                store_means=False,
                store_history=False,
                t0=float(match_times[0]),
            )
            Z[i, j] = float(loglik)

    np.savez_compressed(cache_path, Z=Z, tau_axis=tau_axis, sigma0_axis=sigma0_axis, s=s_fixed)
    return Z


def main():
    # -------------------------
    # Load data once
    # -------------------------
    match_times, match_player_indices, _, id2name, _ = load_wta_numpy()
    n_players = len(id2name)

    match_times = match_times
    match_player_indices = match_player_indices

    # -------------------------
    # Figure 3 axes (article)
    # x = log10(tau) in [-3, -1]
    # y = log10(sigma0^2) in [-2, 0] => sigma0 in [0.1, 1.0]
    # -------------------------
    tau_axis = np.logspace(-3, -0.9, 60)
    sigma0_axis = np.logspace(-1, 0, 60)

    # -------------------------
    # Settings
    # -------------------------
    s_fixed = 1.0

    # Grid should be cheap (fewer particles)
    n_particles_grid = 300

    # EM should be more stable (but still manageable)
    n_particles_em = 500
    n_smooth_traj = 80
    n_em_iters = 100

    # -------------------------
    # 1) Compute / load grid
    # -------------------------
    Z = compute_grid_smc(
        match_times=match_times,
        match_player_indices=match_player_indices,
        n_players=n_players,
        tau_axis=tau_axis,
        sigma0_axis=sigma0_axis,
        s_fixed=s_fixed,
        n_particles_grid=n_particles_grid,
        seed=0,
        cache_path="fig3_smc_grid_cache.npz",
    )

    # -------------------------
    # 2) Run ONE EM path (your init)
    # -------------------------
    init_params = PFParams(sigma0=0.3, tau=0.01, s=s_fixed)

    params_path, _ = run_em_sigma0_tau(
        match_times=match_times,
        match_player_indices=match_player_indices,
        n_players=n_players,
        init_params=init_params,
        n_particles=n_particles_em,
        seed=1,
        n_em_iters=n_em_iters,
        n_smooth_traj=n_smooth_traj,
        player_ids=None,           # all players; for speed you can set np.arange(300)
        t0=float(match_times[0]),
    )

    tau_path = np.array([p.tau for p in params_path], dtype=float)
    sigma0_path = np.array([p.sigma0 for p in params_path], dtype=float)

    # Convert to plot coords: x=log10(tau), y=log10(sigma0^2)
    x_path = np.log10(tau_path)
    y_path = np.log10(sigma0_path ** 2)

    # -------------------------
    # 3) Plot (SMC panel)
    # -------------------------
    X = np.log10(tau_axis)[None, :].repeat(len(sigma0_axis), axis=0)
    Y = np.log10((sigma0_axis ** 2))[:, None].repeat(len(tau_axis), axis=1)

    plt.figure(figsize=(7, 5))
    pcm = plt.pcolormesh(X, Y, Z, shading="auto")
    plt.colorbar(pcm, label="log-likelihood (SMC estimate)")

    plt.plot(x_path, y_path, "-o", linewidth=2)

    plt.xlabel(r"$\log_{10}(\tau)$")
    plt.ylabel(r"$\log_{10}(\sigma_0^2)$")
    plt.title("SMC (grid + one EM path)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
