import numpy as np
from particles import resampling as rs


class PFParams:
    """
    sigma0 : std of initial skill prior  N(0, sigma0^2)
    tau    : RW diffusion parameter: x(t+dt) = x(t) + N(0, tau^2 * dt)
    s      : scale parameter of the logistic likelihood
             P(win) = sigmoid((x_w - x_l) / s)
    """
    def __init__(self, sigma0, tau, s=1.0):
        if s <= 0:
            raise ValueError(f"Parameter s must be strictly positive (got s={s}).")
        if sigma0 <= 0:
            raise ValueError(f"Parameter sigma0 must be strictly positive (got sigma0={sigma0}).")
        if tau <= 0:
            raise ValueError(f"Parameter tau must be strictly positive (got tau={tau}).")
        self.sigma0 = float(sigma0)
        self.tau = float(tau)
        self.s = float(s)


class TennisBootstrap:
    """
    Factorised bootstrap PF:
        - one particle cloud per player i: x[i, :] (size J)
        - at each match (winner, loser): propagate both players to match time,
          weight by match likelihood, resample the pair with the same indices.

    Optional history storage (for smoothing):
        - hist_times[i] : list of times for player i (starts at t0)
        - hist_X[i]     : list of particle arrays (shape (J,)) after resampling
    """

    def __init__(self, params, n_players, n_particles=1000, seed=0, store_history=False, t0=0.0):
        self.params = params
        self.n_players = int(n_players)
        self.n_particles = int(n_particles)
        self.store_history = bool(store_history)

        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)

        # particles for all players: (n_players, J)
        self.x = self.rng.normal(
            loc=0.0,
            scale=self.params.sigma0,
            size=(self.n_players, self.n_particles),
        )

        # last update time per player
        self.last_t = np.full(self.n_players, float(t0), dtype=float)

        # predictive log-likelihood estimate (sum of log mean weights)
        self.loglik = 0.0

        if self.store_history:
            self.hist_times = [[float(t0)] for _ in range(self.n_players)]
            self.hist_X = [[self.x[i].copy()] for i in range(self.n_players)]

    def propagate(self, pid, t):
        """
        Propagate player pid forward to time t under:
            x(t) = x(t_last) + Normal(0, tau^2 * dt),  dt = t - t_last
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
        log G(x_w, x_l) = log sigmoid((x_w - x_l) / s)
        computed in a numerically stable way.
        """
        z = (xw - xl) / self.params.s
        return -np.logaddexp(0.0, -z)

    def store_player_state(self, pid, t):
        self.hist_times[pid].append(float(t))
        self.hist_X[pid].append(self.x[pid].copy())

    # Update for one match
    def update_match(self, t, winner_id, loser_id):
        # 1) propagate both players to t
        self.propagate(winner_id, t)
        self.propagate(loser_id, t)
        xw = self.x[winner_id]
        xl = self.x[loser_id]

        # 2) Log-weights via FK potential
        lw = self.logG(xw, xl)
        wgts = rs.Weights(lw=lw)
        self.loglik += wgts.log_mean

        # 3) resample
        idx = rs.systematic(wgts.W, M=self.n_particles)
        self.x[winner_id] = xw[idx]
        self.x[loser_id] = xl[idx]

        # 4) optional history storage
        if self.store_history:
            self.store_player_state(winner_id, t)
            self.store_player_state(loser_id, t)

    # Loop
    def run(self, match_times, match_player_indices, store_means=False):
        """
        match_times: array-like (K,)
        match_player_indices: array-like (K,2), columns (winner_id, loser_id)

        If store_means=True, returns means_hist of shape (K, n_players):
          means_hist[k, i] = E[x_i | data up to match k]
        """
        match_times = match_times
        match_player_indices = match_player_indices

        means_hist = None
        if store_means:
            means_hist = np.empty((len(match_times), self.n_players), dtype=float)

        for k in range(len(match_times)):
            t = float(match_times[k])
            w = int(match_player_indices[k, 0])
            l = int(match_player_indices[k, 1])
            self.update_match(t, w, l)
            if means_hist is not None:
                means_hist[k] = self.posterior_mean()

        return float(self.loglik), means_hist

    # Results
    def posterior_mean(self):
        return self.x.mean(axis=1)

    def posterior_std(self):
        return self.x.std(axis=1, ddof=0)

    ############
    # Smoothing
    ############

    def smooth_player(self, pid, n_traj=None, seed=None, max_ar_iter=50):
        """
        Returns:
            times: (K_i,) float array
            xs: (K_i, Jt) array of smoothed trajectories samples
        """
        if not self.store_history:
            raise ValueError("Use store_history=True.")

        times = self.hist_times[pid]
        X_list = self.hist_X[pid]
        K = len(X_list)
        if K == 0:
            return times, np.empty((0, 0))

        J = self.n_particles
        Jt = J if n_traj is None else int(n_traj)

        rng = self.rng if seed is None else np.random.default_rng(int(seed))

        # sample terminal states from final filtering particles (uniform)
        XT = X_list[-1]
        idxT = rng.integers(low=0, high=J, size=Jt)
        x_next = XT[idxT].copy()

        xs = np.empty((K, Jt), dtype=float)
        xs[-1, :] = x_next

        tau2 = self.params.tau ** 2

        # backward recursion
        for k in range(K - 2, -1, -1):
            Xk = X_list[k]  # (J,)
            dt = float(times[k + 1] - times[k])
            if dt <= 0.0:
                dt = 1e-12
            var = tau2 * dt

            # accept-reject sampling of ancestor indices for all trajectories
            a = np.full(Jt, -1, dtype=int)
            remaining = np.ones(Jt, dtype=bool)

            # Precompute 1/var
            inv2v = -0.5 / var

            # iterative AR for remaining trajectories
            for _ in range(int(max_ar_iter)):
                if not np.any(remaining):
                    break

                m = np.where(remaining)[0]
                # propose candidates uniformly
                cand = rng.integers(low=0, high=J, size=m.size)

                # acceptance probability
                diff = Xk[cand] - x_next[m]
                log_acc = inv2v * (diff * diff)
                acc = np.exp(log_acc) # in (0,1]
                u = rng.random(m.size)

                accepted = u < acc
                if np.any(accepted):
                    a[m[accepted]] = cand[accepted]
                    remaining[m[accepted]] = False

            # Fallback for rare cases: exact discrete sampling
            if np.any(remaining):
                m = np.where(remaining)[0]
                # compute exact weights for each remaining trajectory separately
                for j in m:
                    diff = Xk - x_next[j]
                    lw = inv2v * (diff * diff)
                    lw -= np.max(lw)
                    w = np.exp(lw)
                    w /= np.sum(w)
                    cdf = np.cumsum(w)
                    a[j] = int(np.searchsorted(cdf, rng.random(), side="right"))

            x_next = Xk[a]
            xs[k, :] = x_next

        return times, xs

    #####
    # EM
    #####

    @staticmethod
    def mstep_sigma0_tau(smoothed_by_player):
        """
        Closed-form M-step updates for sigma0 and tau from smoothed trajectories.

        smoothed_by_player is an iterable of (times_i, xs_i):
            times_i: (K_i,)
            xs_i:    (K_i, Jt)
        """
        num_sigma0 = 0.0
        den_sigma0 = 0

        num_tau = 0.0
        den_tau = 0

        for times_i, xs_i in smoothed_by_player:
            if xs_i.size == 0:
                continue

            K_i, Jt = xs_i.shape

            # sigma0 from initial states
            x0 = xs_i[0, :]
            num_sigma0 += float(np.sum(x0 * x0))
            den_sigma0 += int(Jt)

            # tau from increments
            if K_i >= 2:
                dt = np.diff(times_i)
                dt = np.maximum(dt, 1e-12)
                dx = np.diff(xs_i, axis=0)
                num_tau += float(np.sum((dx * dx) / dt[:, None]))
                den_tau += int((K_i - 1) * Jt)

        sigma0_hat = np.sqrt(num_sigma0 / max(den_sigma0, 1))
        tau_hat = np.sqrt(num_tau / max(den_tau, 1))
        return sigma0_hat, tau_hat




def run_filter(match_times, match_player_indices, n_players, params, n_particles=1000,
                seed=0, store_means=False, store_history=False, t0=0.0,):
    """
    Runs the PF on given data.
    Returns: pf, loglik, means_hist
    """
    match_times = match_times

    pf = TennisBootstrap(
        params=params,
        n_players=n_players,
        n_particles=n_particles,
        seed=seed,
        store_history=store_history,
        t0=t0,
    )
    loglik, means_hist = pf.run(match_times, match_player_indices, store_means=store_means)
    return pf, loglik, means_hist


def run_em_sigma0_tau(match_times, match_player_indices, n_players, init_params, n_particles=1000, seed=0,
                       n_em_iters=10, n_smooth_traj=100, player_ids=None, t0=0.0,):
    """
    EM loop to estimate (sigma0, tau) with s fixed.

    E-step: run PF with store_history=True, then smooth players independently.
    M-step: closed-form updates.

    Returns:
      params_path
    """
    if player_ids is None:
        player_ids = np.arange(n_players, dtype=int)
    else:
        player_ids = player_ids

    params = PFParams(init_params.sigma0, init_params.tau, s=init_params.s)
    params_path = [params]
    loglik_path = []

    for it in range(int(n_em_iters)):
        pf, loglik, _ = run_filter(
            match_times=match_times,
            match_player_indices=match_player_indices,
            n_players=n_players,
            params=params,
            n_particles=n_particles,
            seed=seed + it,
            store_means=False,
            store_history=True,
            t0=t0,
        )
        loglik_path.append(float(loglik))

        smoothed = []
        # smooth each selected player
        for pid in player_ids:
            times_i, xs_i = pf.smooth_player(pid, n_traj=n_smooth_traj, seed=seed + 10_000 + it)
            smoothed.append((times_i, xs_i))

        sigma0_hat, tau_hat = TennisBootstrap.mstep_sigma0_tau(smoothed)
        params = PFParams(sigma0_hat, tau_hat, s=params.s)
        params_path.append(params)

    return params_path, loglik_path
