from data.load_data import load_wta
import jax.numpy as jnp
from models.discrete import Filter
from models.discrete import EM
from scipy.special import logsumexp

#Parameters
match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict = load_wta() 
S = 15
sigma0 = 5
tau = 0.02

#run EM algorithm with discrete method (does not currently work properly)
all_tau, all_sigma0, all_llh = EM(tau=tau,sigma0=sigma0, match_times=match_times, match_player_indices=match_player_indices, players_id_to_name_dict=players_id_to_name_dict, steps=10)
print(all_tau[-1], all_sigma0[-1], all_llh[-1])