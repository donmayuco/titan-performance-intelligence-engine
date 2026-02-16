import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# ==========================================
# ⚙️ CONFIGURATION & CSS
# ==========================================
st.set_page_config(
    page_title="TITAN | PERFORMANCE INTELLIGENCE",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Command Center" Aesthetic
st.markdown("""
<style>
    /* Dark Mode Titan Theme */
    .stApp { background-color: #050911; }
    
    /* Metrics */
    [data-testid="stMetricValue"] { color: #00d2ff !important; font-family: 'Courier New', monospace; }
    [data-testid="stMetricLabel"] { color: #9ca3af !important; }
    
    /* Cards */
    .metric-card {
        background: linear-gradient(145deg, #0f172a, #1e293b);
        padding: 20px;
        border-radius: 8px;
        border-left: 4px solid #00d2ff;
        box-shadow: 0 4px 15px rgba(0,0,0,0.5);
        margin-bottom: 20px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #0b1120; border-right: 1px solid #1e293b; }
    
    /* Inputs */
    .stTextInput > div > div > input { color: #00d2ff; background-color: #0f172a; border: 1px solid #334155; }
    
    /* Login Box */
    .login-box {
        border: 1px solid #334155;
        padding: 40px;
        border-radius: 12px;
        background: rgba(15, 23, 42, 0.8);
        text-align: center;
        max-width: 500px;
        margin: 100px auto;
        box-shadow: 0 0 30px rgba(0, 210, 255, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if 'auth' not in st.session_state:
    st.session_state.auth = False

# ==========================================
# 🔐 THE LOGIN GATE
# ==========================================
def show_login():
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.markdown("""
            <div class="login-box">
                <h1 style='color: #00d2ff;'>🧠 TITAN INTELLIGENCE</h1>
                <h3 style='color: #94a3b8; font-weight: 300;'>PERFORMANCE SEGMENTATION ENGINE</h3>
                <p style='color: #64748b; font-size: 0.8em;'>AUTHORIZED PERSONNEL ONLY</p>
            </div>
        """, unsafe_allow_html=True)
        
        password = st.text_input("ACCESS CODE", type="password", placeholder="Enter Code...", label_visibility="collapsed")
        
        if st.button("INITIALIZE SYSTEM", use_container_width=True):
            if password == "TITAN":
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("⛔ ACCESS DENIED")

# ==========================================
# 🧠 THE ML DATA GENERATOR (Math Engine)
# ==========================================
@st.cache_data
def generate_officer_data(n=500, ita_adoption_rate=0.0):
    np.random.seed(42)
    
    # 1. Create Base Cohorts
    ids = [f"OFC-{1000+i}" for i in range(n)]
    cohorts = np.random.choice(['Rookie (<6mo)', 'Core (6-24mo)', 'Veteran (>2yr)'], n, p=[0.4, 0.4, 0.2])
    data = pd.DataFrame({'Officer_ID': ids, 'Cohort': cohorts})
    
    # 2. Assign Base Stats
    def get_base_stats(cohort):
        if cohort == 'Rookie (<6mo)':
            return np.random.uniform(12, 20), np.random.uniform(65, 80)
        elif cohort == 'Core (6-24mo)':
            return np.random.uniform(8, 14), np.random.uniform(80, 90)
        else:
            return np.random.uniform(4, 8), np.random.uniform(90, 99)

    stats = data['Cohort'].apply(get_base_stats)
    data['Response_Time_Min'] = [x[0] for x in stats]
    data['Protocol_Score'] = [x[1] for x in stats]
    
    # 3. Apply The ITA Engine Effect (Simulation)
    data['Uses_ITA'] = np.random.choice([True, False], n, p=[ita_adoption_rate, 1-ita_adoption_rate])
    
    def apply_ita_boost(row):
        time = row['Response_Time_Min']
        score = row['Protocol_Score']
        if row['Uses_ITA']:
            if row['Cohort'] == 'Rookie (<6mo)':
                time = time * 0.4
                score = score + 15
            else:
                time = time * 0.8
                score = min(score + 5, 100)
        
        time += np.random.normal(0, 0.5)
        score += np.random.normal(0, 1)
        return max(time, 1), min(score, 100)

    data[['Final_Time', 'Final_Score']] = data.apply(apply_ita_boost, axis=1, result_type='expand')
    data['Value_Protected'] = (1000 / data['Final_Time']) * data['Final_Score'] * 10
    
    # --- 🤖 MACHINE LEARNING LAYER (K-MEANS) ---
    X = data[['Final_Time', 'Final_Score']]
    kmeans = KMeans(n_clusters=3, random_state=42)
    data['ML_Cluster'] = kmeans.fit_predict(X)

    # Dynamic Labeling (Teaching the AI what "Good" looks like)
    cluster_labels = {}
    cluster_means = data.groupby('ML_Cluster')['Final_Score'].mean().sort_values()
    sorted_clusters = cluster_means.index.tolist()
    
    cluster_labels[sorted_clusters[0]] = "RISK (Training Needed)"
    cluster_labels[sorted_clusters[1]] = "CORE (Stable)"
    cluster_labels[sorted_clusters[2]] = "ELITE (High Value)"
    
    data['Performance_Segment'] = data['ML_Cluster'].map(cluster_labels)
    
    return data

# ==========================================
# 📊 THE MAIN DASHBOARD (Visual Layer)
# ==========================================
def show_dashboard():
    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        try:
            st.image("titan-logo.png", use_container_width=True)
        except:
            st.header("TITAN OPS")
            
        st.markdown("---")
        st.header("🎮 Simulation Controls")
        
        ita_adoption = st.slider(
            "ITA Engine Adoption (%)", 
            0, 100, 0, 
            help="Drag to see how ML re-classifies Rookies as they improve."
        )
        st.info(f"Simulating {ita_adoption}% of force.")
        
        st.markdown("---")
        st.markdown("### ⚙️ Model Config")
        show_rookies_only = st.checkbox("Focus: Rookies Only", value=False)
        st.caption("Algorithm: K-Means (k=3)")

    # 1. Generate CURRENT Simulation Data (Based on Slider)
    df = generate_officer_data(n=500, ita_adoption_rate=ita_adoption/100)
    
    # 2. Generate BASELINE Data (0% Adoption) for "Real" Deltas
    df_baseline = generate_officer_data(n=500, ita_adoption_rate=0.0)
    
    # Filter Logic (Applies to both Current and Baseline for accurate comparison)
    if show_rookies_only:
        df_display = df[df['Cohort'] == 'Rookie (<6mo)']
        df_base_display = df_baseline[df_baseline['Cohort'] == 'Rookie (<6mo)']
    else:
        df_display = df
        df_base_display = df_baseline

    # --- TOP METRICS (Now with Real Math) ---
    st.title("🧠 Performance Intelligence Engine")
    
    m1, m2, m3, m4 = st.columns(4)
    
    # Calculate Current Stats
    avg_time = df_display['Final_Time'].mean()
    avg_score = df_display['Final_Score'].mean()
    elite_count = len(df_display[df_display['Performance_Segment'] == "ELITE (High Value)"])
    risk_count = len(df_display[df_display['Performance_Segment'].str.contains("RISK")])
    
    # Calculate Baseline Stats (For Deltas)
    base_time = df_base_display['Final_Time'].mean()
    base_score = df_base_display['Final_Score'].mean()
    base_elite = len(df_base_display[df_base_display['Performance_Segment'] == "ELITE (High Value)"])
    base_risk = len(df_base_display[df_base_display['Performance_Segment'].str.contains("RISK")])

    with m1:
        st.metric("Avg Response Time", f"{avg_time:.1f} min", delta=f"{base_time - avg_time:.1f} min faster")
    with m2:
        st.metric("Protocol Precision", f"{avg_score:.1f}%", delta=f"{avg_score - base_score:.1f}% improvement")
    with m3:
        st.metric("Elite Performers", f"{elite_count}", delta=f"+{elite_count - base_elite} officers")
    with m4:
        st.metric("At-Risk Officers", f"{risk_count}", delta=f"{risk_count - base_risk} reduction", delta_color="inverse")

    st.markdown("---")

    # --- THE HYBRID CHART (ML Colors + Efficiency Box) ---
    c1, c2 = st.columns([3, 1])
    
    with c1:
        st.subheader("🎯 Behavioral Clustering (K-Means)")
        st.caption("The AI groups officers by behavior. Green Box = The 'Elite Zone' we target.")
        
        fig = px.scatter(
            df_display,
            x="Final_Time",
            y="Final_Score",
            color="Performance_Segment",
            symbol="Cohort",
            size="Value_Protected",
            hover_data=["Officer_ID", "Uses_ITA"],
            color_discrete_map={
                "ELITE (High Value)": "#00ff41",  # Neon Green
                "CORE (Stable)": "#00d2ff",       # Titan Blue
                "RISK (Training Needed)": "#ff4b4b" # Alert Red
            },
            template="plotly_dark",
            labels={"Final_Time": "Response Time (Minutes)", "Final_Score": "Protocol Accuracy (%)"}
        )
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='#1e293b', range=[0, 25]),
            yaxis=dict(gridcolor='#1e293b', range=[50, 105]),
            height=500,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            shapes=[
                dict(type="rect", x0=0, y0=90, x1=8, y1=105, 
                     line=dict(color="#00ff41", width=2, dash="dot"),
                     fillcolor="rgba(0, 255, 65, 0.1)")
            ]
        )
        fig.add_annotation(x=4, y=102, text="TARGET: ELITE ZONE", showarrow=False, font=dict(color="#00ff41"))
        
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # --- NEW DONUT CHART ---
        st.subheader("📊 Segment Split")
        
        # Prepare Data for Donut
        seg_counts = df_display['Performance_Segment'].value_counts()
        
        # Ensure correct order and handle missing keys if a segment is empty
        labels = ["ELITE (High Value)", "CORE (Stable)", "RISK (Training Needed)"]
        values = [seg_counts.get(l, 0) for l in labels]
        colors = ["#00ff41", "#00d2ff", "#ff4b4b"] # Green, Blue, Red
        
        fig_donut = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.6, # Thick Donut
            sort=False,
            marker=dict(colors=colors, line=dict(color='#000000', width=2)),
            textinfo='percent',
            textfont=dict(size=14, color="white")
        )])
        
        fig_donut.update_layout(
            showlegend=False, 
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=0, b=0, l=0, r=0),
            height=250,
            annotations=[dict(text=f"{len(df_display)}<br>OFC", x=0.5, y=0.5, font_size=20, showarrow=False, font_color="white")]
        )
        
        st.plotly_chart(fig_donut, use_container_width=True)

        # --- EXISTING TABLE ---
        st.markdown("##### 💎 Top Talent")
        gems = df_display[df_display['Performance_Segment'] == "ELITE (High Value)"]
        gems = gems.sort_values('Final_Score', ascending=False).head(5)
        
        if len(gems) > 0:
            st.dataframe(
                gems[['Officer_ID', 'Final_Score']],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Officer_ID": "ID",
                    "Final_Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f%%"),
                }
            )
        else:
            st.info("No Elite officers in this view.")

    # --- FOOTER ---
    st.markdown("---")
    st.caption("SYSTEM STATUS: ONLINE | MODEL: K-MEANS (UNSUPERVISED) | CONNECTION: SECURE")

# ==========================================
# 🚀 APP LAUNCHER
# ==========================================
if st.session_state.auth:
    show_dashboard()
else:
    show_login()

    