from tqdm import tqdm
import jax.random as random
import jax.numpy as jnp

key = random.PRNGKey(0)

def l(skill_diff, s=1):
    return -jnp.log(1+jnp.exp(-skill_diff/ s))

def gl(skill_diff, s=1):
    return 1/(s*(+jnp.exp((skill_diff)/s)))

def hl(skill_diff, s=1):
    return jnp.exp((skill_diff)/s)*(gl(skill_diff, s=s))**2

class ExtendedKalmanFilter():

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
        self.match_times = match_times
        self.match_player_indices = match_player_indices
        self.rankings = jnp.zeros(self.K)

        self.beta = beta
        self.tau = tau
        self.s = s

        self.mu = random.normal(key, shape=(self.N,)) * sigma0
        self.V = jnp.eye(self.N)*sigma0

        self.updated = False

    def update(self, nb_matches=1):


        for match_index in tqdm(range(nb_matches)):

            eps = self.tau**2*(self.match_times[match_index] - self.match_times[match_index-1]) if match_index > 0 else 0

            home_index = self.match_player_indices[match_index][0]
            away_index = self.match_player_indices[match_index][1] 

            V_aux = self.V*self.beta**2 + eps*jnp.eye(self.N)
            
            omega = V_aux[home_index, home_index] + V_aux[away_index, away_index] - 2*V_aux[home_index, away_index]
            g = gl(self.beta*(self.mu[home_index] - self.mu[away_index])/self.s, s=self.s)
            h = hl(self.beta*(self.mu[home_index] - self.mu[away_index])/self.s, s=self.s)

            delta = V_aux[:, home_index] - V_aux[:, away_index]

            self.mu = self.beta * self.mu + (g * self.s) / (self.s**2 + h*omega) * delta
            h = hl(self.beta*(self.mu[home_index] - self.mu[away_index])/self.s, s=self.s)

            self.V = V_aux - (h / (self.s**2 + h*omega)) * jnp.outer(delta, delta) * (h / (self.s**2 + h*omega))

        self.updated = True

    def compute_llh(self):

        if not self.updated:
            self.update(nb_matches=self.K)

        log_likelihood = 0
        for match_index in range(self.K):
            home_index = self.match_player_indices[match_index][0]
            away_index = self.match_player_indices[match_index][1] 
            skill_diff = self.mu[home_index] - self.mu[away_index]
            log_likelihood += l(skill_diff, s=self.s)

        return log_likelihood