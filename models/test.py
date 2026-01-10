from data.load_data_numpy import load_wta_numpy
import numpy as np
from models.discrete import Filter
from scipy.special import logsumexp
match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict = load_wta_numpy() 

S = 5

Psi = np.array([[np.sqrt(2/S)*np.cos(np.pi*i*(2*j-1)/S) for j in range(1,S+1)] for i in range(S)])
Psi[0,:] = np.multiply(np.sqrt(1/2), Psi[0,:])
Psi_inv = np.transpose(Psi)
Lambda_tilde = np.diag([np.cos(np.pi*i/S) for i in range(S)])
Lambda =  Lambda_tilde - np.eye(S)
print(Psi@Psi_inv) 
print(Psi_inv@Psi)

