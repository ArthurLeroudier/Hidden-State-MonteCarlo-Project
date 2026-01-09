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

S = 500

Psi1 = [[1/np.sqrt(S) for i in range(1,S+1)]]
Psi2 = [[np.sqrt(2/S)*np.cos(np.pi-(i-1)*(2*j-1)) for j in range(1,S+1)] for i in range(2, S+1)]
Psi = Psi1 + Psi2
Psi = np.array(Psi)
Psi_inv = np.linalg.inv(Psi)
Lambda =  np.diag([np.cos(2*np.pi*(i-1)) for i in range(1,S+1)]) - np.eye(S)

def times_M(pi, dt, tau):
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

        self.x = [[0.] for pid in n_players] #discrete gaussian centered in S/2, n_players

        self.t = [[0.] for i in range(n_players)]

        self.smoothed = [[] for pid in range(n_players)]


    def propagate(self, pid, t):

        dt = t - self.t[pid][-1]
        Predict_i = times_M(self.x[pid][-1], dt, self.tau)
        self.t[pid].append(t)

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
        f = f/f.sum(axis=1,keepdims=True) #Normalize joint filter row to get distributions

        self.x[winner_id].append(np.transpose(np.dot(f,np.ones(self.S))))
        self.x[loser_id].append(np.dot(np.ones(self.S), f))

        

    # Loop
    def run(self, match_times_arr, match_player_indices_arr):
        means_hist = np.empty((len(match_times_arr), n_players), dtype=float)

        #Initialization of the filtering
        x0 = np.zeros(S)
        x0[S//2] += 1/2
        x0[(S+1)//2] += 1/2
        x0 = times_M(x0, self.sigma0, 1)
        for pid in range(n_players):
            self.x[pid][0] = x0

        for k in range(len(match_times_arr)):
            t = float(match_times_arr[k])
            w_id = int(match_player_indices_arr[k, 0])
            l_id = int(match_player_indices_arr[k, 1])

            self.update_match(t, w_id, l_id)

            means_hist[k] = self.posterior_mean()

        return self.loglik, means_hist    
    
    def posterior_mean(self):
        return self.x.mean(axis=1)

    def smooth_step(self, pid, k):
        dt = self.t[pid][k] - self.t[pid][k+1]
        filter_k = self.x[pid][k]
        smooth = self.smoothed[pid][-1]

        predict = times_M(filter_k, dt, self.tau)
        smooth = np.divide(smooth,predict)
        smooth = np.dot(smooth,np.transpose(Psi))
        e = np.exp(-self.tau*dt*Lambda)
        smooth = np.dot(smooth, e)
        smooth = np.dot(smooth, np.transpose(Psi_inv))
        smooth = np.multiply(filter_k, smooth)

        self.smoothed[pid].append(smooth)

    def smoothing(self):
        for pid in range(n_players):
            K = len(self.x[pid])
            self.smoothed = #INITIALISATION
            for k in range(K-1,-1,-1):
                self.smooth_step(pid,k)
    
    
def new_theta(tau, sigma0):
    filter = Filter(tau= tau, simga0 = sigma0)
    filter.run(match_times_arr, match_player_indices_arr)
    filter.smoothing()
    new_sigma0 = 1/n_players*(sum(filter.smoothed[pid][0] for pid in range(n_players))^2) #empirical initial variance
    Q1 = 

    N = [[] for pid in range(n_players)]
    D = [[] for pid in range(n_players)]

    for pid in range(n_players): #compute necessary quantities for gradient G2 computation
        K = len(filter.x[pid])
        for k in range(1, K+1):
            F_ik = np.dot(filter.x[pid][k-1], Psi_inv)
            S_ik = np.dot(filter.smoothed[pid][k], Psi_inv)

            dt = filter.t[pid][k] - filter.t[pid][k-1]
            e = np.exp(filter.tau * dt * Lambda)
            Lambda_k = dt*np.dot(Lambda,e)

            N_ik = np.dot(F_ik, Lambda_k)
            N_ik = np.dot(N_ik, np.transpose(S_ik))
            D_ik = np.dot(F_ik, Lambda)
            D_ik = np.dot(D_ik, np.transpose(S_ik))

            N[pid].append(N_ik)
            D[pid].append(D_ik)

    dQ2 = sum([sum(N[pid]/D[pid]) for pid in range(n_players)])

def EM(tau, sigma0, steps=1000):
    all_tau = [tau]
    all_sigma0 = [sigma0]
    all_llh = []
    last_tau = tau
    last_sigma0 = sigma0
    for i in steps:
        last_tau, last_sigma0, llh = new_theta(last_tau, last_sigma0)
        all_tau.append(last_tau)
        all_sigma0.append(last_sigma0)
        all_llh.append(llh)