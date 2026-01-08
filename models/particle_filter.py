import numpy as np
from particles import resampling as rs

from data.load_data import load_wta

#############
# Load data #
#############

match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict = load_wta()
n_players = len(players_id_to_name_dict)


##########################
# Fixed model parameters #
##########################

class PFParams:
    """
    sigma0 : std of initial skill prior  N(0, sigma0^2)
    tau    : RW diffusion parameter: x(t+dt) = x(t) + N(0, tau^2 * Delta_t)
    s      : scale parameter of the logistic likelihood
    """
    def __init__(self, sigma0, tau, s=1.0):
        if sigma0 <= 0 or tau <= 0 or s <= 0:
            raise ValueError("All parameters should be positive")
        self.sigma0 = float(sigma0)
        self.tau = float(tau)
        self.s = float(s)


#############
# Bootstrap #
#############

class TennisBootstrapPF:
    def __init__(self, params, n_particles=1000, seed=0):
        self.params = params
        self.n_particles = int(n_particles)

        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)

        # Particles: shape (n_players, n_particles)
        self.x = self.rng.normal(
            loc=0.0,
            scale=params.sigma0,
            size=(n_players, self.n_particles),
        )

        # Last update time per player
        self.last_t = np.zeros(n_players, dtype=float)

        # Predictive log-likelihood estimate
        self.loglik = 0.0

    # Propagate
    def propagate(self, pid, t):
        """
        Docstring for propagate
        
        pid: player id
        t: time of the new match
        M(x_t,x_last) = Normal (x_last, tau^2 (t-t_last))
        """
        dt = float(t) - float(self.last_t[pid])
        if dt > 0.0:
            self.x[pid] += self.rng.normal(
                loc=0.0,
                scale=self.params.tau * np.sqrt(dt),
                size=self.n_particles,
            )
            self.last_t[pid] = float(t)


    def logG(self, xw, xl):
        """
        xw: x of the winner
        wl: x of the loser
        log G(x_w, x_l) = log sigmoid((x_w - x_l) / s)
        """
        z = (xw - xl) / self.params.s
        return -np.logaddexp(0.0, -z)

    # Update for one match
    def update_match(self, t, winner_id, loser_id):
        # 1) Propagate only the two players involved
        self.propagate(winner_id, t)
        self.propagate(loser_id, t)
        x_win = self.x[winner_id]
        x_loser = self.x[loser_id]

        # 2) Log-weights via FK potential
        lw = self.logG(x_win, x_loser)

        # 3) Weights handling (particles-style)
        wgts = rs.Weights(lw=lw)

        # Predictive log-likelihood increment
        self.loglik += wgts.log_mean

        # 4) Pairwise systematic resampling
        idx = rs.systematic(wgts.W, M=self.n_particles)

        self.x[winner_id] = x_win[idx]
        self.x[loser_id] = x_loser[idx]

    # Loop
    def run(self, match_times_arr, match_player_indices_arr):
        means_hist = np.empty((len(match_times_arr), n_players), dtype=float)

        for k in range(len(match_times_arr)):
            t = float(match_times_arr[k])
            w_id = int(match_player_indices_arr[k, 0])
            l_id = int(match_player_indices_arr[k, 1])

            self.update_match(t, w_id, l_id)

            means_hist[k] = self.posterior_mean()

        return self.loglik, means_hist

    # Results
    def posterior_mean(self):
        return self.x.mean(axis=1)

    def posterior_std(self):
        return self.x.std(axis=1)

#####################
# Run the bootstrap #
#####################

def run_wta_filter(params, n_particles=1000, seed=0):
    pf = TennisBootstrapPF(params=params, n_particles=n_particles, seed=seed)
    loglik, means_hist = pf.run(
        match_times,
        match_player_indices
    )
    return pf, loglik, means_hist, players_id_to_name_dict
