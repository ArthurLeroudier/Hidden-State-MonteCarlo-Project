import numpy as np
import matplotlib.pyplot as plt
import numba

import particles
from particles import distributions as dists
from particles import state_space_models as ssms
from particles import resampling as rs
from particles import smoothing

from data.load_data_numpy import load_wta_numpy

match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict = load_wta_numpy()
n_players = len(players_id_to_name_dict)

S = 40

Psi1 = [[1/np.sqrt(S) for i in range(1,S+1)]]
Psi2 = [[np.sqrt(2/S)*np.cos(np.pi-(i-1)*(2*j-1)) for j in range(1,S+1)] for i in range(2, S+1)]
Psi = Psi1 + Psi2
Psi = np.array(Psi)
Psi_inv = np.linalg.inv(Psi)
Lambda =  np.diag([np.cos(2*np.pi*(i-1)) for i in range(1,S+1)]) - np.eye(S)

def times_M(pi, dt, tau):
    #pi is a
    pi1 = np.dot(pi,Psi_inv)
    e = np.exp(-tau*dt*Lambda)
    pi2 = np.dot(pi1, e)
    pi3 = np.dot(pi2, Psi)
    return(pi3)


class Filter():
    def __init__(self,
                 tau, 
                 sigma0,
                 S,
                 s=1):
        self.tau = tau
        self.sigma0 = sigma0
        self.S = S
        self.s = s

        self.x = #discrete gaussian centered in S/2, n_players

        # Last update time per player
        self.last_t = np.zeros(n_players, dtype=float)

    def propagate(self, pid, t):

        dt = t - self.last_t[pid]
        Predict_i = times_M(self.x[pid], dt, self.tau)
        return(Predict_i)
    
    def logG(self, xw, xl):
        """
        xw: x of the winner
        wl: x of the loser
        log G(x_w, x_l) = log sigmoid((x_w - x_l) / s)
        """
        z = (xw - xl) / self.s
        return -np.logaddexp(0.0, -z)

    def update_match(self, t, winner_id, loser_id):
        predict_winner = self.propagate(winner_id, t)
        predict_loser = self.propagate(loser_id, t)
        
        f = np.dot(predict_winner, predict_loser)
        Gk = np.exp(self.logG(predict_winner, predict_loser))
        f = np.multiply(f, Gk)
        #NORMALIZE f
        self.x[winner_id] = np.transpose(np.dot(f,np.ones(self.S)))
        self.x[loser_id] = np.dot(np.ones(self.S), f)

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
    
    def posterior_mean(self):
        return self.x.mean(axis=1)