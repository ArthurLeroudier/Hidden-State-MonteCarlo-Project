import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def gl(z, y):
    pass

def hl(z, y):
    pass

class ExtendedKalmanFilter():

    def __init__(self,
                 match_times,
                 match_player_indices,
                 match_results,
                 tau=1,
                 mu_0 = 0,
                 sigma0 = 1,
                 beta=1
                 ):

        self.K = len(match_results)
        self.match_results = match_results
        self.match_times = match_times
        self.match_player_indices = match_player_indices

        self.beta = beta
        self.tau = tau

        self.mu = mu_0
        self.V = np.eye(self.K)*sigma0

        self.match_index = 0

    def update(self):

        eps = self.tau**2*(self.match_times[self.iteration] - self.match_times[self.iteration-1]) if self.iteration > 0 else 0
        V_aux = self.V*self.beta**2 + eps*np.eye(self.K)

        w = np.matmul()

        pass