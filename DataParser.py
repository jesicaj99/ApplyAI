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
    del hitter_by_game_df['H-AB']
    del hitter_by_game_df['AB']
    del hitter_by_game_df['H']
    del hitter_by_game_df['#P']
    del hitter_by_game_df['Game']
    del hitter_by_game_df['Team']
    del hitter_by_game_df['Hitter Id']
    sorted_hitter_df = hitter_by_game_df.sort_values(by='Hitters')
    return sorted_hitter_df

#gets rids of unnecessary columns in pitcher csv
def clean_sorted_pitcher():
    del pitcher_by_game_df['R']
    del pitcher_by_game_df['ER']
    del pitcher_by_game_df['PC']
    del pitcher_by_game_df['Game']
    del pitcher_by_game_df['Team']
    del pitcher_by_game_df['Extra']
    del pitcher_by_game_df['Pitcher Id']
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
    del warp_hitter_df['bpid']
    del warp_hitter_df['mlbid']
    del warp_hitter_df['Age']
    del warp_hitter_df['DRC+']
    del warp_hitter_df['+/-']
    del warp_hitter_df['PA']
    del warp_hitter_df['AB']
    del warp_hitter_df['R']
    del warp_hitter_df['RBI']
    del warp_hitter_df['SB']
    del warp_hitter_df['AVG']
    del warp_hitter_df['OBP']
    del warp_hitter_df['SLG']
    del warp_hitter_df['ISO']
    del warp_hitter_df['K%']
    del warp_hitter_df['BB%']
    del warp_hitter_df['K/BB']
    del warp_hitter_df['Whiff%']
    sorted_warphitter_df = warp_hitter_df.sort_values(by='WARP')
    return sorted_warphitter_df

def clean_warp_pitcher():
    del warp_pitcher_df['bpid']
    del warp_pitcher_df['mlbid']
    del warp_pitcher_df['team']
    del warp_pitcher_df['DRA-']
    del warp_pitcher_df['DRA']
    del warp_pitcher_df['DRA SD']
    del warp_pitcher_df['cFIP']
    del warp_pitcher_df['G']
    del warp_pitcher_df['GS']
    del warp_pitcher_df['W']
    del warp_pitcher_df['L']
    del warp_pitcher_df['ERA']
    del warp_pitcher_df['RA9']
    del warp_pitcher_df['K%']
    del warp_pitcher_df['BB%']
    del warp_pitcher_df['Whiff%']
    sorted_warppitcher_df = warp_pitcher_df.sort_values(by='WARP')
    return sorted_warppitcher_df

def clean_oaa():
    del oaa_hitter_df['player_id']
    del oaa_hitter_df['display_team_name']
    del oaa_hitter_df['year']
    del oaa_hitter_df['primary_pos_formatted']



