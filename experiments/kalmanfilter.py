import time

from data.load_data import load_wta
from models.kalmanfilters import ExtendedKalmanFilters

match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict = load_wta()
wta_kalman = ExtendedKalmanFilters(match_times,
                                  match_player_indices,
                                  players_id_to_name_dict=players_id_to_name_dict,
                                  tau=0.003,
                                  sigma0 = 0.01,
                                  )
a = time.time()
loglikelihood = wta_kalman.compute_llh(modeltype="DiagonalVariance")
b = time.time()
print("Computation time:", b - a, "seconds")
print("loglikelihood of the WTA data under the Extended Kalman Filter model:", loglikelihood)