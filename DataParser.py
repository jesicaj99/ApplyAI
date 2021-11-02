# Player Game Data Parser
import pandas as pd


hitter_by_game_df = pd.read_csv('hittersByGame(player_offense_data).csv',skiprows=)
pitcher_by_game_df = pd.read_csv('pitchersByGame(pitcher_data).csv')
baserunning_by_game_df = pd.read_csv('baserunningNotes(player_offense_data).csv')
fielding_by_game_df = pd.read_csv('fieldingNotes(player_defensive_data).csv')
warp_hitter_df = pd.read_csv('bp_hitters_2021.csv')
warp_pitcher_df = pd.read_csv('bp_pitchers_2021.csv')
oaa_hitter_df = pd.read_csv('outs_above_average.csv')


#gets rid of unnecessary columns in hitter csv
def clean_sorted_hitter():
    hitter_by_game_df.drop(['H-AB', 'AB', 'H', 'P', 'Game', 'Team', 'Hitter Id'], axis = 1)
    sorted_hitter_df = hitter_by_game_df.sort_values(by='Hitters')
    return sorted_hitter_df

#gets rids of unnecessary columns in pitcher csv
def clean_sorted_pitcher():
    pitcher_by_game_df.drop(['R', 'ER', 'PC', 'Game', 'Team', 'Extra', 'Pitcher Id'], axis = 1)
    sorted_pitcher_df = pitcher_by_game_df.sort_values(by='Pitchers')
    return sorted_pitcher_df

#gets rids of unnecessary fielding columns
def clean_sorted_fielding():
    del fielding_by_game_df['Game']
    sorted_fielding_df = fielding_by_game_df.sort_values(by='Team')
    return sorted_fielding_df

#gets rids of unnececessary baserunning columns
def clean_sorted_baserunning():
    del baserunning_by_game_df['Game']
    sorted_baserunning_df = baserunning_by_game_df.sort_values(by='Team')
    return sorted_baserunning_df

def clean_warp_hitter():
    warp_hitter_df.drop(['bpid', 'mlbid', 'Age', 'DRC+', '+/-', 'PA', 'R', 'RBI','ISO', 'K', 'BB%','Whiff%'], axis = 1)
    sorted_warphitter_df = warp_hitter_df.sort_values(by='WARP')
    return sorted_warphitter_df

def clean_warp_pitcher():
    warp_pitcher_df.drop(['bpid', 'mlbid', 'DRA-', 'DRA+', 'DRA SD', 'cFIP', 'GS', 'W','L', 'ERA', 'RA9','Whiff%'], axis = 1)
    sorted_warppitcher_df = warp_pitcher_df.sort_values(by='WARP')
    return sorted_warppitcher_df

def clean_oaa():
    oaa_hitter_df.drop(['player_id','display_team_name','year','primary_pos_formatted'],axis = 1)


