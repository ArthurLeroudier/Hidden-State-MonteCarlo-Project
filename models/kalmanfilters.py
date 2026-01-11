from tqdm import tqdm
import jax
import jax.random as random
import jax.numpy as jnp
from jax import lax

key = random.PRNGKey(0)

def l(skill_diff, s=1):
    return -jnp.log(1+jnp.exp(-skill_diff/ s))

def gl(skill_diff, s=1):
    return 1/(s*(+jnp.exp((skill_diff)/s)))

def hl(skill_diff, s=1):
    aux = gl(skill_diff, s=s)
    return jnp.exp((skill_diff)/s)*aux*aux

class ExtendedKalmanFilters():

    def __init__(self,
                 match_times,
                 match_player_indices,
                 players_id_to_name_dict=None,
                 tau=1,
                 sigma0 = 1,
                 beta=1,
                 s=1
                 ):

        self.K = len(match_player_indices)
        self.N = len(players_id_to_name_dict)
        self.match_times = jnp.array(match_times)
        self.match_player_indices = jnp.array(match_player_indices)
        self.rankings = jnp.zeros(self.K)

        time_deltas = jnp.diff(self.match_times, prepend=self.match_times[0])
        self.matches_info = jnp.column_stack([self.match_player_indices, time_deltas])

        self.beta = beta
        self.s = s

        self.tau = tau
        self.sigma0 = sigma0

        self.mu = random.normal(key, shape=(self.N,)) * sigma0
        self.V = jnp.eye(self.N)*sigma0

        self.log_likelihood = None

        self.updated = False

    def filtering(self, modelType = "FullVariance"):

        if modelType == "FullCovariance":

            mu0 = self.mu
            V0 = self.V
            loglikelihood0 = 0.0

            def step(carry, info):
                mu, V, loglikelihood = carry
                home, away, eps = info

                V_aux = self.beta**2 * V + eps * jnp.eye(self.N)

                omega = V_aux[home, home] + V_aux[away, away] - 2 * V_aux[home, away]

                skill_diff = mu[home] - mu[away]
                diff = self.beta * skill_diff / self.s

                loglikelihood = loglikelihood + l(skill_diff, s=self.s)

                g = gl(diff, s=self.s)
                h = hl(diff, s=self.s)

                denom = self.s**2 + h * omega
                coeff = (g * self.s) / denom
                coeff2 = h / denom

                delta = V_aux[:, home] - V_aux[:, away]

                mu = self.beta * mu + coeff * delta
                V = V_aux - coeff2 * jnp.outer(delta, delta)

                return (mu, V, loglikelihood), (mu, jnp.diag(V))

            (mu_final, V_final, loglikelihood_final), (mu_hist, v_hist) = jax.lax.scan(
                step, (mu0, V0, loglikelihood0), self.matches_info
            )

            self.mu = mu_final
            self.V = V_final
            self.log_likelihood = loglikelihood_final
            self.mu_hist = mu_hist.T
            self.v_hist = v_hist.T
            self.updated = True


        if modelType == "DiagonalVariance":

            mu0 = self.mu
            v0 = jnp.full(self.N, self.sigma0**2)

            def step(carry, inp):
                mu, v, loglikelihood = carry
                home, away, dt = inp

                eps = (self.tau ** 2) * dt
                V_aux = (self.beta**2) * v + eps

                Vh = V_aux[home]
                Va = V_aux[away]
                omega = Vh + Va

                skill_diff = mu[home] - mu[away]
                diff = self.beta * skill_diff / self.s

                loglikelihood = loglikelihood + l(skill_diff, s=self.s)

                g = gl(diff, s=self.s)
                h = hl(diff, s=self.s)

                denom = self.s**2 + h * omega
                coeff = (g * self.s) / denom
                coeff2 = h / denom

                mu = mu * self.beta
                mu = mu.at[home].add(coeff * Vh)
                mu = mu.at[away].add(-coeff * Va)

                v = v.at[home].set(Vh * (1 - coeff2 * Vh))
                v = v.at[away].set(Va * (1 - coeff2 * Va))

                return (mu, v, loglikelihood), (mu, v)

            (mu_final, v_final, loglikelihood_final), (mu_hist, v_hist) = jax.lax.scan(step, (mu0, v0, 0.0), self.matches_info)

            self.mu = mu_final
            self.V = jnp.diag(v_final)
            self.log_likelihood = loglikelihood_final
            self.mu_hist = mu_hist.T
            self.v_hist = v_hist.T
            self.updated = True

    
