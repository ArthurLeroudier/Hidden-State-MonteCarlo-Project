import numpy as np
from tqdm import tqdm

from data.load_data_numpy import load_wta_numpy
from models.smc import PFParams, run_filter, run_em_sigma0_tau


def compute_grid_smc(match_times, match_player_indices, n_players, tau_axis, sigma0_axis, s_fixed=1.0, n_particles_grid=300, seed=0, cache_path="data/smc_cache.npz",
):
    """
    Compute Z[i,j] = loglik(tau_axis[j], sigma0_axis[i]) with PF.
    Save data in data/
    """

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

    np.savez_compressed(
        cache_path,
        Z=Z,
        tau_axis=tau_axis,
        sigma0_axis=sigma0_axis,
        s=s_fixed,
        n_particles_grid=n_particles_grid,
        seed=seed,
    )
    return Z


def main():
    match_times, match_player_indices, _, id2name, _ = load_wta_numpy()
    n_players = len(id2name)

    tau_axis = np.logspace(-3, -0.9, 60)
    sigma0_axis = np.logspace(-1, 0, 60)

    s_fixed = 1.0
    n_particles_grid = 300

    n_particles_em = 400
    n_smooth_traj = 80
    n_em_iters = 150

    # Grid
    Z = compute_grid_smc(
        match_times=match_times,
        match_player_indices=match_player_indices,
        n_players=n_players,
        tau_axis=tau_axis,
        sigma0_axis=sigma0_axis,
        s_fixed=s_fixed,
        n_particles_grid=n_particles_grid,
        seed=0,
        cache_path="data/smc_cache.npz",
    )

    # EM Path
    init_params = PFParams(sigma0=0.3, tau=0.01, s=s_fixed)

    params_path, loglik_path = run_em_sigma0_tau(
        match_times=match_times,
        match_player_indices=match_player_indices,
        n_players=n_players,
        init_params=init_params,
        n_particles=n_particles_em,
        seed=1,
        n_em_iters=n_em_iters,
        n_smooth_traj=n_smooth_traj,
        player_ids=None,
        t0=float(match_times[0]),
    )

    tau_path = np.array([p.tau for p in params_path], dtype=float)
    sigma0_path = np.array([p.sigma0 for p in params_path], dtype=float)

    # Plot coords: x=log10(tau), y=log10(sigma0^2)
    x_path = np.log10(tau_path)
    y_path = np.log10(sigma0_path ** 2)

    # Save
    out_path = "data/fig3_smc_data.npz"
    np.savez_compressed(
        out_path,
        tau_axis=tau_axis,
        sigma0_axis=sigma0_axis,
        Z=Z,
        tau_path=tau_path,
        sigma0_path=sigma0_path,
        x_path=x_path,
        y_path=y_path,
        loglik_path=np.asarray(loglik_path, dtype=float),
        s_fixed=float(s_fixed),
        n_particles_grid=int(n_particles_grid),
        n_particles_em=int(n_particles_em),
        n_smooth_traj=int(n_smooth_traj),
        n_em_iters=int(n_em_iters),
        init_sigma0=float(init_params.sigma0),
        init_tau=float(init_params.tau),
        seed_grid=int(0),
        seed_em=int(1),
    )

    print(f"Saved data to: {out_path}")


if __name__ == "__main__":
    main()
