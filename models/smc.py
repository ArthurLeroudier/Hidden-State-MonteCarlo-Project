import numpy as np
import matplotlib.pyplot as plt
import numba

import particles
from particles import distributions as dists
from particles import state_space_models as ssms
from particles import resampling as rs
from particles import smoothing

from data.load_data import load_wta

match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict = load_wta()

n_matches = len(match_results)
n_players = len(list(players_id_to_name_dict.keys()))

class Tennis_skill(ssms.StateSpaceModel):
    default_params = (tau,
                      sigma0,
                      mu0 = 0,
                      s=1
                      )

    def PX0(self):
        return dists.Normal(self.mu0, np.square(self.sigma0))

    def PX(self, t, prev_t, prev_x):
        return dists.Normal(prev_x, (t-prev_t)*self.tau^2)

    def PY(self, x, x_opp):
        return dists.Logistic(loc= x - x_opp)



class SMCFilter():

    def __init__(self,
                 match_times,
                 match_player_indices,
                 players_id_to_name_dict=None,
                 tau=1,
                 sigma0 = 1,
                 mu0 = 0,
                 s=1
                 ):
        
        self.K = len(match_player_indices)
        self.N = len(players_id_to_name_dict)
        self.match_times = match_times
        self.match_player_indices = match_player_indices

        self.tau = tau
        self.sigma0 = sigma0

        self.mu0 = 0
        self.s = s
    
    def smoothing(tau, sigma0, data, N=1000):
        fk = ssms.Bootstrap(ssm=Tennis_skill(tau=tau, sigma0= sigma0), data=data)
        filter = particles.SMC(fk=fk, N=N, qmc=False, store_history=True)
        filter.run()
        paths = filter.hist.backward_sampling_reject(N, max_trials==N*10**6)
        return (paths, filter.logLt)

    def new_theta(tau, sigma0, J=1000):
        paths, llh = self.smoothing(tau, sigma0, N=J)
        new_tau = sum(x for x in )
        new_sigma0 = sum
        return (new_tau, new_sigma0, llh)


    def EM(tau, sigma0, N=1000, steps = 1000):
        all_tau = [tau]
        all_sigma0 = [sigma0]
        all_llh = []
        last_tau = tau
        last_sigma0 = sigma0
        for i in steps:
            last_tau, last_sigma0, llh = self.new_theta(last_tau, last_sigma0, N=1000)
            all_tau.append(last_tau)
            all_sigma0.append(last_sigma0)
            all_llh.append(llh)
        return (all_tau, all_sigma0, all_llh)

