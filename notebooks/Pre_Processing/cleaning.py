import pandas as pd
import numpy as np

# ---------------------------------------------------------
# STEP 1: LOAD DATA
# ---------------------------------------------------------
print("Step 1: Loading CSV...")
file_path = r"C:\Users\lenovo\MLops Project\IPL dataset\IPL.csv"

try:
    df = pd.read_csv(file_path)
    print(f"✅ Loaded! Original Rows: {len(df)}")
except FileNotFoundError:
    print("❌ ERROR: File nahi mili path par check kar.")
    exit()

# ---------------------------------------------------------
# STEP 2: DATE FIXING
# ---------------------------------------------------------
print("Step 2: Fixing Dates...")
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

# ---------------------------------------------------------
# STEP 3 & 4: STANDARDIZE TEAMS (CRITICAL FIX ✅)
# ---------------------------------------------------------
print("Step 3 & 4: Standardizing Team Names (Batting, Bowling, Winner)...")

# 1. Major Teams Mapping (Consistent Names)
team_mapping = {
    # Delhi
    'Delhi Daredevils': 'Delhi Capitals',
    
    # Punjab
    'Kings XI Punjab': 'Punjab Kings',
    
    # Bangalore
    'Royal Challengers Bangalore': 'Royal Challengers Bengaluru',
    
    # Hyderabad (Deccan Chargers -> SRH)
    'Deccan Chargers': 'Sunrisers Hyderabad'
}

# 2. Spelling Corrections (For consistency)
spelling_fix = {
    'Rising Pune Supergiants': 'Rising Pune Supergiant', 
    'Pune Warriors': 'Pune Warriors India'
}

# 3. Apply Mappings to ALL Team Columns
cols_to_fix = ['batting_team', 'bowling_team', 'match_won_by', 'toss_winner']

for col in cols_to_fix:
    if col in df.columns:
        df[col] = df[col].replace(team_mapping)
        df[col] = df[col].replace(spelling_fix)

print("✅ Teams Fixed! Unique Winners Check:")
print(df['match_won_by'].dropna().unique())

# ---------------------------------------------------------
# STEP 0 (Moved here): SORTING
# ---------------------------------------------------------
print("Sorting Data (Match > Innings > Over > Ball)...")
df = df.sort_values(['match_id', 'innings', 'over', 'ball'])

# ---------------------------------------------------------
# STEP 5: CLEANING & DROPPING
# ---------------------------------------------------------
print("Step 5: Filtering Bad Matches & Dropping Columns...")

# 1. Remove 'No Result' Matches
if 'result_type' in df.columns:
    initial_rows = len(df)
    
    # NaN results usually mean 'Normal' match, so fill them first
    df['result_type'] = df['result_type'].fillna('normal')
    
    # Filter out 'no result' or 'abandoned'
    df = df[~df['result_type'].str.lower().isin(['no result', 'abandoned'])]
    
    print(f"Removed {initial_rows - len(df)} 'No Result' rows.")

# 2. Drop Unnecessary Columns
cols_to_drop = [
    'Unnamed: 0', 'event_name', 'match_type', 'win_outcome', 
    'team_type', 'overs', 'balls_per_over', 'superover_winner', 
    'umpire', 'player_of_match', 'gender', 'match_number', 
    'next_batter', 'review_decision', 'team_reviewed', 'review_batter'
    
]

existing_cols = [c for c in cols_to_drop if c in df.columns]
df = df.drop(existing_cols, axis=1)

print(f"Data Cleaned. Shape: {df.shape}")

# ---------------------------------------------------------
# STEP 6: DETERMINE FIRST TO BAT
# ---------------------------------------------------------
print("Step 6: Calculating 'First to Bat'...")

first_bat_mapping = df[df['innings'] == 1].groupby('match_id')['batting_team'].first()
df['first_to_bat'] = df['match_id'].map(first_bat_mapping)

# Drop Toss Columns 
df = df.drop(['toss_decision', 'toss_winner'], axis=1, errors='ignore')

# ---------------------------------------------------------
# STEP 7: SUMMARIES (Now with Clean Team Names ✅)
# ---------------------------------------------------------
print("Step 7: Creating Summaries...")

# 7.1 MATCHES (High Level Info)
ball_cols = [
    'over', 'ball', 'ball_no', 'batter', 'bat_pos', 'runs_batter',
    'balls_faced', 'bowler', 'valid_ball', 'runs_extras',
    'runs_total', 'runs_bowler', 'runs_not_boundary', 'extra_type',
    'non_striker', 'non_striker_pos', 'wicket_kind', 'player_out',
    'fielders', 'umpires_call', 'event_match_no', 'new_batter',
    'batter_runs', 'batter_balls', 'bowler_wicket', 'batting_partners',
    'striker_out'
]
matches = df.drop(ball_cols, axis=1, errors='ignore').drop_duplicates(subset=['match_id'])

# 7.2 INNING WISE SUMMARY
matches_inning_wise = df.groupby(['match_id', 'innings']).agg({
    'batting_team': 'first',
    'bowling_team': 'first',
    'first_to_bat': 'first',
    'runs_target': 'first',
    'match_won_by': 'first',
    'team_runs': 'last',    
    'team_balls': 'last',
    'team_wicket': 'last'
}).reset_index()

# 7.3 MATCH SUMMARY (Team A vs Team B)
summary = matches_inning_wise.groupby('match_id').agg({
    'batting_team': 'first', # Inning 1 team (Team A)
    'bowling_team': 'first', # Inning 1 bowling team (Team B)
    'runs_target': 'last',   # Target (Usually set in Inning 2 row)
    'match_won_by': 'first'
}).rename(columns={'batting_team': 'team_a', 'bowling_team': 'team_b'})

# Add Scores and Wickets
summary['team_a_runs'] = matches_inning_wise[matches_inning_wise['innings']==1].set_index('match_id')['team_runs']
summary['team_b_runs'] = matches_inning_wise[matches_inning_wise['innings']==2].set_index('match_id')['team_runs']
summary['team_a_wickets'] = matches_inning_wise[matches_inning_wise['innings']==1].set_index('match_id')['team_wicket']
summary['team_b_wickets'] = matches_inning_wise[matches_inning_wise['innings']==2].set_index('match_id')['team_wicket']

summary = summary.reset_index()

# ---------------------------------------------------------
# STEP 8: PLAYER STATISTICS
# ---------------------------------------------------------
print("Step 8: Calculating Player Stats...")

# 8.1 Batting Stats
batters_matchwise = df.groupby(['match_id', 'batter']).agg({
    'runs_batter': 'sum',
    'balls_faced': 'sum'
}).reset_index()

batters = batters_matchwise.groupby('batter').agg({
    'runs_batter': 'sum',
    'balls_faced': 'sum'
}).reset_index()
batters['batting_avg'] = batters['runs_batter'] / batters['balls_faced'] # Runs per Ball

# 8.2 Bowling Stats
bowlers_matchwise = df.groupby(['match_id', 'bowler']).agg({
    'runs_bowler': 'sum',
    'bowler_wicket': 'sum'
}).reset_index()

# Valid Balls Count (For Economy)
if 'valid_ball' in df.columns:
    balls_bowled_count = df.groupby(['match_id', 'bowler'])['valid_ball'].sum().reset_index(name='balls_bowled')
else:
    balls_bowled_count = df.groupby(['match_id', 'bowler']).size().reset_index(name='balls_bowled')

bowlers_matchwise = bowlers_matchwise.merge(balls_bowled_count, on=['match_id', 'bowler'])

bowlers = bowlers_matchwise.groupby('bowler').agg({
    'runs_bowler': 'sum',
    'balls_bowled': 'sum'
}).reset_index()

# Economy Calculation
bowlers['economy'] = (bowlers['runs_bowler'] * 6) / bowlers['balls_bowled']
bowlers['economy'] = bowlers['economy'].fillna(bowlers['economy'].mean())

# 8.3 Map Stats to Main DF
df['batter_avg'] = df['batter'].map(batters.set_index('batter')['batting_avg'])
df['econ'] = df['bowler'].map(bowlers.set_index('bowler')['economy'])

df['batter_avg'] = df['batter_avg'].fillna(df['batter_avg'].mean())
df['econ'] = df['econ'].fillna(df['econ'].mean())

# ---------------------------------------------------------
# STEP 9: ROLLING FEATURES
# ---------------------------------------------------------
print("Step 9: Calculating Rolling Features...")

# Wicket Fix
if 'wicket_kind' in df.columns:
    df['wicket_kind'] = df['wicket_kind'].fillna('not_out')
    df['is_wicket'] = df['wicket_kind'].apply(lambda x: 0 if x == 'not_out' else 1)
else:
    df['is_wicket'] = df['bowler_wicket']

groups = df.groupby(['match_id', 'innings'])
df['runs_last_6'] = groups['runs_total'].transform(lambda x: x.rolling(6).sum().fillna(0))
df['runs_last_12'] = groups['runs_total'].transform(lambda x: x.rolling(12).sum().fillna(0))
df['wkts_last_12'] = groups['is_wicket'].transform(lambda x: x.rolling(12).sum().fillna(0))

# ---------------------------------------------------------
# STEP 10: SAVE ALL FILES
# ---------------------------------------------------------
print("Step 10: Saving Final Files...")

# Clean columns before saving
cols_to_remove = ['runs_bowler', 'new_batter', 'runs_batter', 'runs_extras', 'valid_ball']
df_final = df.drop([c for c in cols_to_remove if c in df.columns], axis=1)

innings_1 = df_final[df_final['innings'] == 1]
innings_2 = df_final[df_final['innings'] == 2]

# Saving
df_final.to_csv("data_final_features.csv", index=False)
innings_1.to_csv("innings_1.csv", index=False)
innings_2.to_csv("innings_2.csv", index=False)

batters.to_csv("batters.csv", index=False)
bowlers.to_csv("bowlers.csv", index=False)
batters_matchwise.to_csv("batters_matchwise.csv", index=False)
bowlers_matchwise.to_csv("bowlers_matchwise.csv", index=False)

matches.to_csv("matches.csv", index=False)
matches_inning_wise.to_csv("matches_inning_wise.csv", index=False)
summary.to_csv("matches_summary.csv", index=False)

print("🎉 Pura Pipeline Complete! All files are consistent and saved.")