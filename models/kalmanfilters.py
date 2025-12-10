from jax import numpy as jnp


def l(skill_diff, s=1):
    return -jnp.log(1+jnp.exp(-skill_diff/ s))

def gl(skill_diff, s=1):
    return 1/(s*(+jnp.exp((skill_diff)/s)))

def hl(skill_diff, s=1):
    return -jnp.exp((skill_diff)/s)*(gl(skill_diff, s=s))**2

class ExtendedKalmanFilter():

    def __init__(self,
                 match_times,
                 match_player_indices,
                 tau=1,
                 mu_0 = 0,
                 sigma0 = 1,
                 beta=1,
                 s=1
                 ):

        self.K = len(match_player_indices)
        self.N = jnp.max(match_player_indices)
        self.match_times = match_times
        self.match_player_indices = match_player_indices
        self.rankings = jnp.zeros(self.K)

        self.beta = beta
        self.tau = tau
        self.s = s

        self.mu = mu_0
        self.V = jnp.eye(self.K)*sigma0

        self.match_index = 0

    def update(self):

        eps = self.tau**2*(self.match_times[self.match_index] - self.match_times[self.match_index-1]) if self.match_index > 0 else 0

        home_index = self.match_player_indices[self.match_index][0]
        away_index = self.match_player_indices[self.match_index][1] 

        V_aux = self.V*self.beta**2 + eps*jnp.eye(self.K)

        omega = V_aux[home_index, home_index] + V_aux[away_index, away_index] - 2*V_aux[home_index, away_index]
        g = gl(self.beta*(self.mu[home_index] - self.mu[away_index])/self.s, s=self.s)
        h = hl(self.beta*(self.mu[home_index] - self.mu[away_index])/self.s, s=self.s)

        self.mu = self.beta*self.mu + (g*self.s)/(self.s**2+ h*omega)*jnp.array([V_aux[i, home_index] - V_aux[i, away_index] for i in range(self.K)])
        h = hl(self.beta*(self.mu[home_index] - self.mu[away_index])/self.s, s=self.s)

        self.V = V_aux - h/(self.s**2+h*omega)*jnp.array([[(V_aux[i, home_index] - V_aux[i, away_index])*(V_aux[j, home_index] - V_aux[j, away_index])*h/(self.s**2 + h*omega) for j in range(self.K)] for i in range(self.K)])
        self.match_index += 1

    def compute_likelihood(self):
        log_likelihood = 0
        for match_index in range(self.K):
            home_index = self.match_player_indices[match_index][0]
            away_index = self.match_player_indices[match_index][1] 
            skill_diff = self.mu[home_index] - self.mu[away_index]
            log_likelihood += l(skill_diff, s=self.s)
        return log_likelihood