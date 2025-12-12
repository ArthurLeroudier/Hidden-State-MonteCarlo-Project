from jax import numpy as jnp
import matplotlib.pyplot as plt
import numba

import particles
from particles import distributions as dists
from particles import state_space_models as ssm  

from data.load_data import load_wta

match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict = load_wta()

n_matches = len(match_results)
n_players = len(list(players_id_to_name_dict.keys()))

