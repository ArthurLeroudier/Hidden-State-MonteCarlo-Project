import numpy as np
import pandas as pd

from data.load_data import load_wta

match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict = load_wta()
print("Match times:", match_times)
print("Match player indices:", match_player_indices)
print("Match results:", match_results)
print("Players ID to Name dict:", players_id_to_name_dict)
print("Players Name to ID dict:", players_name_to_id_dict)
print("Number of matches loaded:", len(match_times))