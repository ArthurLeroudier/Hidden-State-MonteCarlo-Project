from data.load_data_numpy import load_wta_numpy
import numpy as np
from models.discrete import Filter
from models.discrete import EM
from scipy.special import logsumexp
match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict = load_wta_numpy() 

S = 10
sigma0 = 0.01
tau = 0.001


"""f = Filter(tau=tau, sigma0=sigma0,S=S)
f.run(match_times=match_times, match_player_indices=match_player_indices)
llh = f.smoothing(match_times=match_times, match_players_indices=match_player_indices)"""

all_tau, all_sigma0, all_llh = EM(tau=tau,sigma0=sigma0, match_times=match_times, match_player_indices=match_player_indices, steps=100)
print(all_tau[-1], all_sigma0[-1], all_llh[-1])