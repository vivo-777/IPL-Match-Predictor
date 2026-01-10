import streamlit as st
import pickle
import pandas as pd
import numpy as np
import xgboost as xgb
import time
import warnings
from sklearn.preprocessing import LabelEncoder
import os
os.environ["XGBOOST_DISABLE_GPU"] = "1"
class ContextAwareModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=10,
            tree_method="hist",
            device="cpu",
            n_jobs=-1,
            random_state=42
        )
        self.encoders = {}
        self.feature_cols = [
            'venue', 'batting_team', 'bowling_team',
            'batter', 'bowler', 'over', 'innings', 'team_wicket'
        ]
        self.teams_per_year_dynamic = {}

    def load_and_train(self):
        inn1 = pd.read_csv('dataset/innings_1.csv')
        inn2 = pd.read_csv('dataset/innings_2.csv')
        df = pd.concat([inn1, inn2])
        df = df.loc[:, ~df.columns.duplicated()]

        if 'venue' not in df.columns:
            df['venue'] = df.get('city', 'Unknown')

        df['venue'] = df['venue'].astype(str)

        if 'season' in df.columns:
            df['year'] = df['season'].astype(str).str[:4].astype(int)
        else:
            df['year'] = pd.to_datetime(df['date']).dt.year

        df['team_wicket'] = df.get('team_wicket', 0)
        df.dropna(subset=['batter', 'bowler', 'runs_total', 'over'], inplace=True)

        for col in ['venue', 'batting_team', 'bowling_team', 'batter', 'bowler']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.encoders[col] = le

        df['over'] = df['over'].astype(int)
        df['innings'] = df['innings'].astype(int)
        df['team_wicket'] = df['team_wicket'].astype(int)

        y_raw = np.where(df['is_wicket'] == 1, -1, df['runs_total'])
        self.target_encoder = LabelEncoder()
        y = self.target_encoder.fit_transform(y_raw)

        self.model.fit(df[self.feature_cols], y)
        self.df = df

        temp = df[['year', 'batting_team']]
        temp['team_name'] = self.encoders['batting_team'].inverse_transform(temp['batting_team'])
        for y in temp['year'].unique():
            self.teams_per_year_dynamic[y] = sorted(
                temp[temp['year'] == y]['team_name'].unique()
            )

    def get_squad(self, team_name, year):
        team_le = self.encoders['batting_team']
        if team_name not in team_le.classes_:
            return [], None, team_name

        team_id = team_le.transform([team_name])[0]
        subset = self.df[(self.df['year'] == year) & (self.df['batting_team'] == team_id)]

        if subset.empty:
            return [], team_id, team_name

        top_bats = subset.groupby('batter')['runs_total'].sum().nlargest(7).index.tolist()
        subset_bowl = self.df[(self.df['year'] == year) & (self.df['bowling_team'] == team_id)]
        top_bowls = subset_bowl[subset_bowl['is_wicket'] == 1] \
            .groupby('bowler')['is_wicket'].sum().nlargest(5).index.tolist()

        squad_ids = list(set(top_bats + top_bowls))[:11]
        squad_names = self.encoders['batter'].inverse_transform(squad_ids)
        return list(zip(squad_names, squad_ids)), team_id, team_name

    def predict_ball(self, input_vec):
        probs = self.model.predict_proba(input_vec.astype(np.float32))[0]
        idx = np.random.choice(len(probs), p=probs)
        return self.target_encoder.inverse_transform([idx])[0]

# Warnings ignore
warnings.filterwarnings('ignore')

# ------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ------------------------------------------------------------------
st.set_page_config(
    page_title="IPL Super App 🏏",
    page_icon="🏆",
    layout="wide"
)
from constants import teams_per_year_static, model_translation, cities

# --- CACHE ---
@st.cache_resource
def load_simulator():
    with open("model/context_simulator.pkl", "rb") as f:
        sim = pickle.load(f)
    return sim

def force_xgb_cpu_pipeline(pipe):
    if pipe is None:
        return None

    for step in pipe.named_steps.values():
        if hasattr(step, "get_booster"):
            booster = step.get_booster()

            # HARD force CPU
            booster.set_param("device", "cpu")
            booster.set_param("predictor", "cpu_predictor")
            booster.set_param("tree_method", "hist")

# Load Pickle Models
try:
    pipe_2nd = pickle.load(open('model/pipe_2nd_innings.pkl', 'rb'))
    force_xgb_cpu_pipeline(pipe_2nd)
except: pipe_2nd = None

try:
    pipe_1st = pickle.load(open('model/pipe_1st_innings.pkl', 'rb'))
    force_xgb_cpu_pipeline(pipe_1st)
except: pipe_1st = None



# Init
with st.spinner("⏳ Starting AI Engine"):
    simulator = load_simulator()

# ------------------------------------------------------------------
# 5. UI LOGIC
# ------------------------------------------------------------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Mode", 
    ["🔮 Win Predictor (2nd Innings)", "📊 Score Predictor (1st Innings)", "⚔️ Fantasy Match Simulation"])

# ==================================================================
# MODE 1: FANTASY SIMULATION
# ==================================================================
if app_mode == "⚔️ Fantasy Match Simulation":
    st.header("⚔️ Fantasy Match Simulator")
    
    # Init Session State
    if 'sim_result' not in st.session_state:
        st.session_state.sim_result = None

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Team 1")
        avail_years = list(simulator.teams_per_year_dynamic.keys())
        y1 = st.selectbox("Year (Team 1)", avail_years, index=0, key='y1')
        teams_y1 = simulator.teams_per_year_dynamic.get(y1, [])
        t1 = st.selectbox("Team 1", teams_y1, key='t1')
        
    with col2:
        st.subheader("Team 2")
        y2 = st.selectbox("Year (Team 2)", avail_years, index=1 if len(avail_years)>1 else 0, key='y2')
        teams_y2 = simulator.teams_per_year_dynamic.get(y2, [])
        t2 = st.selectbox("Team 2", teams_y2, key='t2')
        
    venue = st.selectbox("Venue", ["Wankhede Stadium", "M Chinnaswamy Stadium", "MA Chidambaram Stadium", "Eden Gardens", "Arun Jaitley Stadium"])
    iterations = st.slider("Simulation Accuracy (Iterations)", 10, 100, 20)
    
    if st.button("🚀 Run Simulation"):
        sq1, id1, n1 = simulator.get_squad(t1, y1)
        sq2, id2, n2 = simulator.get_squad(t2, y2)
        
        try:
            vid = simulator.encoders['venue'].transform([venue])[0]
        except:
            vid = simulator.encoders['venue'].transform(['Wankhede Stadium'])[0]
            
        if not sq1 or not sq2:
            st.error("Error: Squad data missing.")
        else:
            # SHOW PROGRESS
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            t1_wins = 0
            scores1, scores2 = [], []
            
            # LOOP
            for i in range(iterations):
                progress_bar.progress((i + 1) / iterations)
                status_text.text(f"⚡ Simulating Match {i+1}...")
                
                # Logic (CPU optimized)
                r1, w1 = 0, 0
                st_idx, nst_idx = 0, 1
                batters, bowlers = [x[1] for x in sq1], [x[1] for x in sq2]
                bowl_cycle = [bowlers[k % len(bowlers)] for k in range(20)]
                
                # Innings 1
                for over in range(20):
                    if w1 >= 10: break
                    curr_bowler = bowl_cycle[over]
                    for ball in range(6):
                        if w1 >= 10: break
                        inp = np.array([[vid, id1, id2, batters[st_idx], curr_bowler, over, 1, w1]])
                        out = simulator.predict_ball(inp)
                        if out == -1:
                            w1 += 1
                            st_idx = max(st_idx, nst_idx) + 1
                            if st_idx >= len(batters): break
                        else:
                            r1 += out
                            if out % 2 != 0: st_idx, nst_idx = nst_idx, st_idx
                    st_idx, nst_idx = nst_idx, st_idx
                
                # Innings 2
                r2, w2 = 0, 0
                st_idx, nst_idx = 0, 1
                batters, bowlers = [x[1] for x in sq2], [x[1] for x in sq1]
                bowl_cycle = [bowlers[k % len(bowlers)] for k in range(20)]
                
                for over in range(20):
                    if w2 >= 10 or r2 > r1: break
                    curr_bowler = bowl_cycle[over]
                    for ball in range(6):
                        if w2 >= 10 or r2 > r1: break
                        inp = np.array([[vid, id2, id1, batters[st_idx], curr_bowler, over, 2, w2]])
                        out = simulator.predict_ball(inp)
                        if out == -1:
                            w2 += 1
                            st_idx = max(st_idx, nst_idx) + 1
                            if st_idx >= len(batters): break
                        else:
                            r2 += out
                            if out % 2 != 0: st_idx, nst_idx = nst_idx, st_idx
                    st_idx, nst_idx = nst_idx, st_idx
                
                scores1.append(r1)
                scores2.append(r2)
                if r1 > r2: t1_wins += 1
            
            # SAVE TO STATE
            st.session_state.sim_result = {
                'n1': n1, 'n2': n2,
                'win_pct_1': int((t1_wins / iterations) * 100),
                'avg_1': int(np.mean(scores1)),
                'avg_2': int(np.mean(scores2))
            }
            status_text.empty()
            st.rerun()

    # --- DISPLAY RESULTS (Persistent) ---
    if st.session_state.sim_result:
        res = st.session_state.sim_result
        st.markdown("---")
        st.subheader("🏆 Simulation Results")
        
        c1, c2 = st.columns(2)
        c1.metric(label=f"Avg Score: {res['n1']}", value=f"{res['avg_1']}", delta=f"Win Prob: {res['win_pct_1']}%")
        c2.metric(label=f"Avg Score: {res['n2']}", value=f"{res['avg_2']}", delta=f"Win Prob: {100-res['win_pct_1']}%")
        
        if res['win_pct_1'] > 50:
            st.success(f"🏅 **{res['n1']}** wins the simulation!")
        else:
            st.success(f"🏅 **{res['n2']}** wins the simulation!")
        
        if st.button("Reset"):
            st.session_state.sim_result = None
            st.rerun()

# ==================================================================
# MODE 2: 2ND INNINGS PREDICTOR
# ==================================================================
elif app_mode == "🔮 Win Predictor (2nd Innings)":
    st.header("🔮 2nd Innings Win Probability")

    selected_year = st.selectbox(
        'Select Season',
        sorted(teams_per_year_static.keys(), reverse=True)
    )

    current_teams = sorted(teams_per_year_static[selected_year])

    col1, col2 = st.columns(2)
    with col1:
        batting_team = st.selectbox('Batting Team', current_teams)
    with col2:
        bowling_team = st.selectbox('Bowling Team', current_teams)

    selected_city = st.selectbox('City', sorted(cities))

    target = st.number_input('Target Score', min_value=1)

    col3, col4, col5 = st.columns(3)
    score = col3.number_input('Current Score', min_value=0)
    overs = col4.number_input('Overs', min_value=0.0, max_value=19.5, step=0.1)
    wickets = col5.number_input('Wickets', min_value=0, max_value=9)

    if st.button('Predict'):
        if batting_team != bowling_team:
            if pipe_2nd:
                runs_left = target - score
                balls_left = 120 - (int(overs) * 6 + int((overs % 1) * 10))
                wickets_left = 10 - wickets

                crr = score / overs if overs > 0 else 0
                rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

                input_df = pd.DataFrame({
                    'batting_team': [model_translation.get(batting_team, batting_team)],
                    'bowling_team': [model_translation.get(bowling_team, bowling_team)],
                    'city': [selected_city],
                    'runs_needed': [runs_left],
                    'balls_left': [balls_left],
                    'wickets_left': [wickets_left],
                    'crr': [crr],
                    'rrr': [rrr]
                })

                probs = pipe_2nd.predict_proba(input_df)[0]

                bat_prob = probs[1] * 100
                bowl_prob = probs[0] * 100

                st.markdown("### 🏏 Win Probability")

                # ---- METRICS ----
                m1, m2 = st.columns(2)
                m1.metric(
                    label=f"{batting_team}",
                    value=f"{bat_prob:.1f}%",
                    delta="Batting"
                )
                m2.metric(
                    label=f"{bowling_team}",
                    value=f"{bowl_prob:.1f}%",
                    delta="Bowling"
                )

                st.markdown("---")

                # ---- PROGRESS BAR ----
                st.markdown(
                    f"""
                    <div style="width:100%; background:#eee; border-radius:8px; overflow:hidden;">
                        <div style="
                            width:{bat_prob}%;
                            background:#4CAF50;
                            padding:8px 0;
                            float:left;
                            text-align:center;
                            color:white;
                            font-weight:bold;">
                            {bat_prob:.1f}%
                        </div>
                        <div style="
                            width:{bowl_prob}%;
                            background:#f44336;
                            padding:8px 0;
                            float:left;
                            text-align:center;
                            color:white;
                            font-weight:bold;">
                            {bowl_prob:.1f}%
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                # ---- TEXT VERDICT ----
                st.markdown("---")
                if bat_prob > bowl_prob:
                    st.success(f"🏆 **{batting_team}** are more likely to win")
                else:
                    st.success(f"🏆 **{bowling_team}** are more likely to win")

            else:
                st.error("Model Missing")
        else:
            st.error("Batting and Bowling teams cannot be same")

# ==================================================================
# MODE 3: 1ST INNINGS PREDICTOR
# ==================================================================
elif app_mode == "📊 Score Predictor (1st Innings)":
    st.header("📊 1st Innings Score Predictor")
    selected_year = st.selectbox('Select Season', sorted(teams_per_year_static.keys(), reverse=True))
    current_teams = sorted(teams_per_year_static[selected_year])
    col1, col2 = st.columns(2)
    with col1: batting_team = st.selectbox('Batting Team', current_teams)
    with col2: bowling_team = st.selectbox('Bowling Team', current_teams)
    selected_city = st.selectbox('City', sorted(cities))
    col3, col4, col5, col6 = st.columns(4)
    curr_score = col3.number_input('Current Score', min_value=0)
    overs = col4.number_input('Overs', min_value=0.0, max_value=20.0, step=0.1)
    wickets = col5.number_input('Wickets', min_value=0, max_value=9)
    last_five = col6.number_input('Runs Last 5 Overs', min_value=0)
    
    if st.button('Predict'):
        if batting_team == bowling_team:
            st.error("Batting and bowling team cannot be same")
        elif curr_score<last_five:
            st.error("Current Score cannot be less than Score in the last 5 overs")
        else:
            if pipe_1st:
                balls_left = 120 - (int(overs)*6 + int((overs%1)*10))
                wickets_left = 10 - wickets
                crr = curr_score/overs if overs > 0 else 0
                input_df = pd.DataFrame({'batting_team':[model_translation.get(batting_team, batting_team)], 
                                        'bowling_team':[model_translation.get(bowling_team, bowling_team)], 
                                        'city':[selected_city], 'current_score':[curr_score], 
                                        'balls_left':[balls_left], 'wickets_left':[wickets], 
                                        'crr':[crr], 'runs_last_5':[last_five], "year":[selected_year]})
                res = pipe_1st.predict(input_df)
                st.success(f"Projected Score: {int(res[0])}")
            else: st.error("Model Missing")
