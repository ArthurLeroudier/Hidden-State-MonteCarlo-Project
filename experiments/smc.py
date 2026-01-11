import numpy as np
from tqdm import tqdm

from data.load_data_numpy import load_wta_numpy
from models.smc import PFParams, run_filter, run_em_sigma0_tau

import argparse


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


def parse_args():
    p = argparse.ArgumentParser(description="Run SMC grid + EM and save outputs.")
    p.add_argument("--n_particles_grid", type=int, default=300, help="Particles for grid PF.")
    p.add_argument("--n_particles_em", type=int, default=400, help="Particles for EM PF.")
    p.add_argument("--n_smooth_traj", type=int, default=80, help="Number of smoothing trajectories.")
    p.add_argument("--n_em_iters", type=int, default=150, help="EM iterations.")
    p.add_argument("--seed_grid", type=int, default=0, help="Seed for grid PF.")
    p.add_argument("--seed_em", type=int, default=1, help="Seed for EM PF.")
    p.add_argument("--cache_path", type=str, default="data/smc_cache.npz", help="Grid cache path.")
    p.add_argument("--out_path", type=str, default="data/fig3_smc_data.npz", help="Output NPZ path.")
    p.add_argument("--s_fixed", type=float, default=1.0, help="Fixed s parameter.")
    p.add_argument("--n_tau", type=int, default=60, help="Number of tau grid points.")
    p.add_argument("--n_sigma0", type=int, default=60, help="Number of sigma0 grid points.")
    p.add_argument("--init_sigma0", type=float, default=0.3, help="Initial value of sigma0 for the EM")
    p.add_argument("--init_tau", type=float, default=0.01, help="Initial value of tau for the EM")
    return p.parse_args()


def main():
    args = parse_args()

    match_times, match_player_indices, _, id2name, _ = load_wta_numpy()
    n_players = len(id2name)

    tau_axis = np.logspace(-3, -0.9, args.n_tau)
    sigma0_axis = np.logspace(-1, 0, args.n_sigma0)

    # Grid
    Z = compute_grid_smc(
        match_times=match_times,
        match_player_indices=match_player_indices,
        n_players=n_players,
        tau_axis=tau_axis,
        sigma0_axis=sigma0_axis,
        s_fixed=args.s_fixed,
        n_particles_grid=args.n_particles_grid,
        seed=args.seed_grid,
        cache_path=args.cache_path,
    )

    # EM Path
    init_params = PFParams(sigma0=args.init_sigma0, tau=args.init_tau, s=args.s_fixed)

    params_path, loglik_path = run_em_sigma0_tau(
        match_times=match_times,
        match_player_indices=match_player_indices,
        n_players=n_players,
        init_params=init_params,
        n_particles=args.n_particles_em,
        seed=args.seed_em,
        n_em_iters=args.n_em_iters,
        n_smooth_traj=args.n_smooth_traj,
        player_ids=None,
        t0=float(match_times[0]),
    )

    tau_path = np.array([p.tau for p in params_path], dtype=float)
    sigma0_path = np.array([p.sigma0 for p in params_path], dtype=float)

    x_path = np.log10(tau_path)
    y_path = np.log10(sigma0_path ** 2)

    np.savez_compressed(
        args.out_path,
        tau_axis=tau_axis,
        sigma0_axis=sigma0_axis,
        Z=Z,
        tau_path=tau_path,
        sigma0_path=sigma0_path,
        x_path=x_path,
        y_path=y_path,
        loglik_path=np.asarray(loglik_path, dtype=float),
        s_fixed=float(args.s_fixed),
        n_particles_grid=int(args.n_particles_grid),
        n_particles_em=int(args.n_particles_em),
        n_smooth_traj=int(args.n_smooth_traj),
        n_em_iters=int(args.n_em_iters),
        init_sigma0=float(init_params.sigma0),
        init_tau=float(init_params.tau),
        seed_grid=int(args.seed_grid),
        seed_em=int(args.seed_em),
    )

    print(f"Saved data to: {args.out_path}")



if __name__ == "__main__":
    main()
