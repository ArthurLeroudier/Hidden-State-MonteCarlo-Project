from data.load_data import load_wta

length = 10

match_times, match_player_indices, match_results, players_id_to_name_dict, players_name_to_id_dict = load_wta()
print("Match times:", match_times[:length])
print("Match player indices:", match_player_indices[:length])
print("Match results:", match_results[:length])
print("Players ID to Name dict:", players_id_to_name_dict)
print("Players Name to ID dict:", players_name_to_id_dict)
print("Number of matches loaded:", len(match_times))
print("Number of players loaded:", len(players_id_to_name_dict))

for i in range(len(match_results)): # :))))))))))
    if match_results[i] != 1:
        print("Error in match results at index", i)
        break