from data.load_data import load_wta
from models.kalmanfilters import ExtendedKalmanFilter

match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict = load_wta()
wta_kalman = ExtendedKalmanFilter(match_times,
                                  match_player_indices,
                                  players_id_to_name_dict=players_id_to_name_dict,
                                  tau=1,
                                  sigma0 = 1
                                  )
likelihood = wta_kalman.compute_likelihood()
print("likelihood of the WTA data under the Extended Kalman Filter model:", likelihood)