import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib
import os

@st.cache_resource
def load_model():
    return joblib.load("rf_cv_model.pkl")

# Load the trained model
model_path = "random_forest_model.pkl"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    st.error("Model file not found.")

st.set_page_config(
    page_title="🌱 PlantPulse AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --sage:      #4a7c59;
    --mint:      #a8d5b5;
    --foam:      #e8f5ee;
    --leaf:      #2d6a4f;
    --sprout:    #74c69d;
    --sand:      #f5f0e8;
    --bark:      #3d2b1f;
    --sun:       #f4a261;
    --sky:       #90caf9;
    --red:       #e57373;
    --white:     #ffffff;
    --text:      #1b3a2d;
    --muted:     #6b8f71;
    --border:    rgba(74,124,89,0.15);
    --shadow:    0 4px 24px rgba(45,106,79,0.10);
    --shadow-lg: 0 8px 40px rgba(45,106,79,0.15);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
.stApp {
    background: linear-gradient(160deg, #f0f7f2 0%, #e8f5ee 40%, #f5f0e8 100%);
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

.dash-header {
    background: linear-gradient(135deg, var(--leaf) 0%, var(--sage) 60%, var(--sprout) 100%);
    border-radius: 24px;
    padding: 36px 48px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow-lg);
}
.dash-header::before {
    content: '🌿';
    position: absolute;
    right: 48px; top: 50%;
    transform: translateY(-50%);
    font-size: 100px;
    opacity: 0.18;
}
.dash-header h1 {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    color: white !important;
    margin: 0 0 6px;
    line-height: 1.1;
}
.dash-header p {
    color: rgba(255,255,255,0.75);
    font-size: 1rem;
    margin: 0;
    font-weight: 300;
}

.card {
    background: var(--white);
    border-radius: 20px;
    padding: 24px 26px;
    margin-bottom: 20px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: var(--shadow-lg);
}
.card-title {
    font-family: 'DM Serif Display', serif;
    font-size: 1.15rem;
    color: var(--leaf);
    margin: 0 0 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.kpi-tile {
    background: var(--white);
    border-radius: 18px;
    padding: 22px 24px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
    position: relative;
    overflow: hidden;
    transition: transform 0.2s;
    height: 100%;
}
.kpi-tile:hover { transform: translateY(-3px); }
.kpi-tile::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 5px; height: 100%;
    border-radius: 18px 0 0 18px;
}
.kpi-tile.green::before  { background: var(--sage); }
.kpi-tile.blue::before   { background: var(--sky); }
.kpi-tile.amber::before  { background: var(--sun); }
.kpi-tile.red::before    { background: var(--red); }
.kpi-tile.mint::before   { background: var(--sprout); }
.kpi-tile.bark::before   { background: #8d6e63; }

.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
}
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem;
    color: var(--text);
    line-height: 1;
}
.kpi-unit {
    font-size: 0.85rem;
    color: var(--muted);
    font-weight: 400;
    margin-left: 3px;
}
.kpi-sub {
    font-size: 0.78rem;
    color: var(--muted);
    margin-top: 6px;
}
.kpi-icon {
    position: absolute;
    right: 18px; top: 18px;
    font-size: 1.8rem;
    opacity: 0.22;
}

.section-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem;
    color: var(--leaf);
    margin: 32px 0 16px;
    border-left: 4px solid var(--sprout);
    padding-left: 14px;
}

.green-divider {
    height: 2px;
    background: linear-gradient(90deg, var(--sprout), transparent);
    border: none;
    margin: 8px 0 24px;
    border-radius: 2px;
}

/* Insight boxes */
.insight-box {
    border-radius: 14px;
    padding: 16px 20px;
    margin-bottom: 12px;
    border-left: 4px solid;
    font-size: 0.88rem;
    line-height: 1.6;
}
.insight-green {
    background: #d1fae5;
    border-color: #059669;
    color: #064e3b;
}
.insight-amber {
    background: #fef3c7;
    border-color: #d97706;
    color: #78350f;
}
.insight-red {
    background: #fee2e2;
    border-color: #dc2626;
    color: #7f1d1d;
}
.insight-blue {
    background: #eff6ff;
    border-color: #3b82f6;
    color: #1e3a5f;
}
.insight-title {
    font-weight: 700;
    font-size: 0.92rem;
    margin-bottom: 4px;
}

/* Range indicator */
.range-row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 10px;
    font-size: 0.84rem;
}
.range-label { color: var(--text); font-weight: 500; min-width: 120px; }
.range-bar-wrap { flex: 1; margin: 0 12px; }
.range-bar-bg {
    height: 8px; border-radius: 99px;
    background: var(--foam); overflow: hidden;
}
.range-bar-fill { height: 100%; border-radius: 99px; }
.range-value { color: var(--muted); min-width: 60px; text-align: right; }

/* Threshold badge */
.thresh-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 600;
}
.thresh-ok    { background:#d1fae5; color:#065f46; }
.thresh-warn  { background:#fef3c7; color:#78350f; }
.thresh-alert { background:#fee2e2; color:#991b1b; }

.stButton > button {
    background: linear-gradient(135deg, var(--leaf), var(--sage)) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 12px 36px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    letter-spacing: 0.03em !important;
    box-shadow: 0 4px 16px rgba(45,106,79,0.30) !important;
    transition: all 0.2s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1a4d36, var(--leaf)) !important;
    box-shadow: 0 6px 24px rgba(45,106,79,0.40) !important;
    transform: translateY(-2px) !important;
}

[data-baseweb="slider"] [role="slider"] {
    background: var(--sage) !important;
    border-color: var(--sage) !important;
}

.dash-footer {
    text-align: center;
    padding: 24px;
    color: var(--muted);
    font-size: 0.78rem;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY THEME ──────────────────────────────────────────────────────────────
CHART_LAYOUT = dict(
    template="plotly_white",
    font_family="DM Sans",
    font_color="#1b3a2d",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=20, r=20, t=40, b=20),
    title_font_family="DM Serif Display",
    title_font_size=15,
    title_font_color="#2d6a4f",
)
GREEN_SEQ = ["#2d6a4f","#4a7c59","#74c69d","#a8d5b5","#d1fae5"]

# ─── OPTIMAL RANGES (for human-readable insights) ─────────────────────────────
OPTIMAL = {
    "Temperature_C":       (18, 28,  "°C",   "Most plants thrive at 18–28°C"),
    "Humidity_%":          (50, 80,  "%",    "50–80% humidity keeps leaves hydrated"),
    "Soil_Moisture_%":     (30, 70,  "%",    "30–70% soil moisture avoids root rot & drought"),
    "Soil_pH":             (5.5, 7.0,"",     "pH 5.5–7.0 unlocks nutrient uptake"),
    "Nutrient_Level":      (50, 90,  "",     "Aim for nutrient level 50–90"),
    "Light_Intensity_lux": (8000, 25000, " lux", "8k–25k lux for active growth"),
}

# ─── DATA & MODEL ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    # Try augmented dataset first, fall back to original
    try:
        d = pd.read_csv("plant_data_augmented.csv").dropna()
    except FileNotFoundError:
        d = pd.read_csv("plant_moniter_health_data.csv").dropna()
    d["Soil_Moisture_%"] = d["Soil_Moisture_%"].abs()
    d.index = pd.RangeIndex(len(d))
    return d

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

df    = load_data()
model = load_model()

# ─── FEATURE SETUP ─────────────────────────────────────────────────────────────
FEATURES    = ["Temperature_C","Humidity_%","Soil_Moisture_%","Soil_pH","Nutrient_Level","Light_Intensity_lux"]
FEAT_LABELS = ["Temperature","Humidity","Soil Moisture","Soil pH","Nutrient Level","Light Intensity"]
importance  = model.feature_importances_

# ─── STATUS MAP ────────────────────────────────────────────────────────────────
STATUS_MAP = {1:"Healthy", 0:"At Risk", "Healthy":"Healthy", "Unhealthy":"At Risk"}
df2 = df.copy()
df2["Status"] = df2["Health_Status"].map(STATUS_MAP).fillna(df2["Health_Status"].astype(str))

health_counts = df["Health_Status"].value_counts()
total         = len(df)
healthy_pct   = round(health_counts.get(1, health_counts.get("Healthy", 0)) / total * 100, 1)
atrisk_pct    = round(100 - healthy_pct, 1)

# Dataset source tag
try:
    pd.read_csv("plant_data_augmented.csv")
    data_source = "Augmented Dataset (incl. Synthetic)"
except FileNotFoundError:
    data_source = "Original Dataset"

# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-header">
  <h1>🌱 PlantPulse AI</h1>
  <p>Smart Plant Health Monitoring · {total:,} records · {data_source} · Random Forest Model Active</p>
</div>
""", unsafe_allow_html=True)

# ─── KPI TILES ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpis = [
    (k1, "green",  "🌡", "Avg Temperature",  f"{df['Temperature_C'].mean():.1f}", "°C",  f"Max {df['Temperature_C'].max():.0f}°C"),
    (k2, "blue",   "💧", "Avg Humidity",      f"{df['Humidity_%'].mean():.1f}",   "%",   f"Min {df['Humidity_%'].min():.0f}%"),
    (k3, "mint",   "🪴", "Soil Moisture",     f"{df['Soil_Moisture_%'].mean():.1f}", "%", f"Std {df['Soil_Moisture_%'].std():.1f}"),
    (k4, "amber",  "☀️", "Avg Light",         f"{df['Light_Intensity_lux'].mean()/1000:.1f}", "klux", f"Max {df['Light_Intensity_lux'].max()/1000:.0f}k"),
    (k5, "bark",   "🧪", "Avg Nutrient",      f"{df['Nutrient_Level'].mean():.1f}", "",   f"Min {df['Nutrient_Level'].min():.0f}"),
    (k6, "red",    "🌿", "Healthy Plants",    f"{healthy_pct}", "%",              f"of {total:,} records"),
]
for col, color, icon, label, val, unit, sub in kpis:
    with col:
        st.markdown(f"""
        <div class="kpi-tile {color}">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{val}<span class="kpi-unit">{unit}</span></div>
          <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

# ─── SECTION 1: HEALTH OVERVIEW ────────────────────────────────────────────────
st.markdown('<div class="section-heading">Plant Health Overview</div>', unsafe_allow_html=True)
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1.2, 1.2, 1.6], gap="large")

with col_a:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌿 Health Status Distribution</div>', unsafe_allow_html=True)
    hc = df["Health_Status"].value_counts().reset_index()
    hc.columns = ["Status","Count"]
    hc["Label"] = hc["Status"].map(STATUS_MAP).fillna(hc["Status"].astype(str))
    fig = go.Figure(go.Pie(
        labels=hc["Label"], values=hc["Count"],
        hole=0.62,
        marker_colors=["#4a7c59","#e57373"],
        textinfo="percent",
        textfont_size=13,
        hovertemplate="%{label}: %{value} plants<extra></extra>"
    ))
    fig.add_annotation(text=f"<b>{healthy_pct}%</b><br><span style='font-size:11px'>Healthy</span>",
                       x=0.5, y=0.5, showarrow=False, font_size=18, font_color="#2d6a4f")
    fig.update_layout(**CHART_LAYOUT, showlegend=True,
                      legend=dict(orientation="h", y=-0.05),
                      height=280)
    st.plotly_chart(fig, use_container_width=True)
    # Plain-English summary
    emoji = "✅" if healthy_pct >= 70 else ("⚠️" if healthy_pct >= 50 else "🚨")
    st.markdown(f"""
    <div class="insight-box {'insight-green' if healthy_pct>=70 else ('insight-amber' if healthy_pct>=50 else 'insight-red')}">
      <div class="insight-title">{emoji} What this means</div>
      {healthy_pct}% of your plants are healthy. {atrisk_pct}% show signs of stress — 
      {'Great job! Keep monitoring for early alerts.' if healthy_pct>=70 else 'Attention needed. Check soil, water, and light conditions.' }
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_b:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧪 Nutrient Level Spread</div>', unsafe_allow_html=True)
    bins   = pd.cut(df["Nutrient_Level"], bins=[0,30,60,80,101], labels=["Low (0–30)","Medium (30–60)","Good (60–80)","High (80+)"])
    ndf    = bins.value_counts().sort_index().reset_index()
    ndf.columns = ["Range","Count"]
    fig = go.Figure(go.Pie(
        labels=ndf["Range"], values=ndf["Count"],
        hole=0.55,
        marker_colors=["#e57373","#f4a261","#74c69d","#2d6a4f"],
        textinfo="percent+label",
        textfont_size=11,
        hovertemplate="%{label}: %{value} plants<extra></extra>"
    ))
    fig.update_layout(**CHART_LAYOUT, showlegend=False, height=280)
    st.plotly_chart(fig, use_container_width=True)
    low_pct = round((df["Nutrient_Level"] < 30).sum() / total * 100, 1)
    st.markdown(f"""
    <div class="insight-box {'insight-red' if low_pct>20 else 'insight-green'}">
      <div class="insight-title">💡 Nutrient Insight</div>
      {low_pct}% of plants have low nutrients (&lt;30). 
      {'Consider fertilising — low nutrients directly link to plant stress.' if low_pct>20 else 'Nutrient levels look well distributed across the dataset.'}
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_c:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🤖 What the AI Looks At (Feature Importance)</div>', unsafe_allow_html=True)
    fi_df = pd.DataFrame({"Feature": FEAT_LABELS, "Importance": importance}).sort_values("Importance")
    fig = go.Figure(go.Bar(
        x=fi_df["Importance"], y=fi_df["Feature"],
        orientation="h",
        marker=dict(
            color=fi_df["Importance"],
            colorscale=[[0,"#a8d5b5"],[0.5,"#4a7c59"],[1,"#1a4d36"]],
            showscale=False
        ),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        text=[f"{v:.3f}" for v in fi_df["Importance"]],
        textposition="outside",
        textfont_size=11,
    ))
    fig.update_layout(**CHART_LAYOUT, xaxis_title="Importance Score",
                      yaxis_title="", height=230,
                      xaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    top_feat = fi_df.iloc[-1]["Feature"]
    st.markdown(f"""
    <div class="insight-box insight-blue">
      <div class="insight-title">🧠 Model Logic</div>
      <b>{top_feat}</b> is the most influential factor in predicting plant health. 
      Higher bar = AI relies more on that sensor reading when making its decision.
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── SECTION 2: ENVIRONMENT HEALTH CHECK ──────────────────────────────────────
st.markdown('<div class="section-heading">📋 Dataset Health Check — Are Conditions Optimal?</div>', unsafe_allow_html=True)
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-title">🔎 How Each Sensor Reading Compares to Ideal Plant Conditions</div>', unsafe_allow_html=True)
st.markdown("*Green = in optimal range · Orange = borderline · Red = out of range*", unsafe_allow_html=False)

check_cols = st.columns(3)
feat_check_items = [
    ("Temperature_C",       "🌡 Temperature"),
    ("Humidity_%",          "💧 Humidity"),
    ("Soil_Moisture_%",     "🪴 Soil Moisture"),
    ("Soil_pH",             "⚗️ Soil pH"),
    ("Nutrient_Level",      "🧪 Nutrient Level"),
    ("Light_Intensity_lux", "☀️ Light Intensity"),
]

for i, (feat, label) in enumerate(feat_check_items):
    lo, hi, unit, tip = OPTIMAL[feat]
    avg = df[feat].mean()
    in_range_pct = round(((df[feat] >= lo) & (df[feat] <= hi)).sum() / total * 100, 1)
    
    if in_range_pct >= 65:
        badge_cls, badge_txt = "thresh-ok", "✅ Good"
    elif in_range_pct >= 40:
        badge_cls, badge_txt = "thresh-warn", "⚠️ Watch"
    else:
        badge_cls, badge_txt = "thresh-alert", "🚨 Alert"

    col = check_cols[i % 3]
    with col:
        fill_color = "#4a7c59" if in_range_pct>=65 else ("#f4a261" if in_range_pct>=40 else "#e57373")
        st.markdown(f"""
        <div style="margin-bottom:18px;">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="font-weight:600;font-size:0.88rem;">{label}</span>
            <span class="thresh-badge {badge_cls}">{badge_txt}</span>
          </div>
          <div style="font-size:0.78rem;color:#6b8f71;margin-bottom:6px;">
            Avg: <b>{avg:.1f}{unit}</b> · Optimal: {lo}–{hi}{unit} · <b>{in_range_pct}% in range</b>
          </div>
          <div class="range-bar-bg">
            <div class="range-bar-fill" style="width:{in_range_pct}%;background:{fill_color};"></div>
          </div>
          <div style="font-size:0.76rem;color:#6b8f71;margin-top:4px;font-style:italic;">{tip}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─── SECTION 3: ENVIRONMENT TRENDS ────────────────────────────────────────────
st.markdown('<div class="section-heading">Environment Trends Over Time</div>', unsafe_allow_html=True)
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

window = max(5, len(df)//100)
df_roll = df[FEATURES].rolling(window, min_periods=1).mean()

col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌡 Temperature & Humidity (Smoothed Trend)</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df_roll["Temperature_C"], name="Temperature °C",
        line=dict(color="#e57373", width=2.5),
        fill="tozeroy", fillcolor="rgba(229,115,115,0.08)",
        hovertemplate="Temp: %{y:.1f}°C<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        y=df_roll["Humidity_%"], name="Humidity %",
        line=dict(color="#90caf9", width=2.5),
        fill="tozeroy", fillcolor="rgba(144,202,249,0.08)",
        hovertemplate="Humidity: %{y:.1f}%<extra></extra>"
    ))
    # Optimal range shading
    fig.add_hrect(y0=18, y1=28, fillcolor="rgba(74,124,89,0.06)",
                  line_width=0, annotation_text="Ideal Temp", annotation_position="top left",
                  annotation_font_size=10, annotation_font_color="#4a7c59")
    fig.update_layout(**CHART_LAYOUT, height=260,
                      legend=dict(orientation="h", y=-0.15),
                      xaxis=dict(showgrid=False, title="Record Index"),
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🪴 Soil Moisture & pH (Smoothed Trend)</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df_roll["Soil_Moisture_%"], name="Soil Moisture %",
        line=dict(color="#74c69d", width=2.5),
        fill="tozeroy", fillcolor="rgba(116,198,157,0.12)",
        hovertemplate="Moisture: %{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        y=df_roll["Soil_pH"]*10, name="Soil pH (×10 scaled)",
        line=dict(color="#f4a261", width=2.5, dash="dot"),
        hovertemplate="pH: %{customdata:.1f}<extra></extra>",
        customdata=df_roll["Soil_pH"]
    ))
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(74,124,89,0.06)",
                  line_width=0, annotation_text="Ideal Moisture", annotation_position="top left",
                  annotation_font_size=10, annotation_font_color="#4a7c59")
    fig.update_layout(**CHART_LAYOUT, height=260,
                      legend=dict(orientation="h", y=-0.15),
                      xaxis=dict(showgrid=False, title="Record Index"),
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── SECTION 4: DISTRIBUTIONS ─────────────────────────────────────────────────
st.markdown('<div class="section-heading">Feature Distributions</div>', unsafe_allow_html=True)
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")
for col, feat, label, color in [
    (col1, "Temperature_C",       "🌡 Temperature (°C)",    "#e57373"),
    (col2, "Soil_Moisture_%",     "💧 Soil Moisture (%)",   "#74c69d"),
    (col3, "Light_Intensity_lux", "☀️ Light Intensity (lux)","#f4a261"),
]:
    with col:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f'<div class="card-title">{label}</div>', unsafe_allow_html=True)
        fig = px.histogram(df, x=feat, nbins=30, color_discrete_sequence=[color],
                           labels={feat: label})
        fig.update_traces(marker_line_width=0.5, marker_line_color="white", opacity=0.85)
        lo, hi, unit, _ = OPTIMAL[feat]
        fig.add_vrect(x0=lo, x1=hi, fillcolor="rgba(74,124,89,0.12)",
                      line_width=1, line_color="#4a7c59",
                      annotation_text="Ideal", annotation_position="top left",
                      annotation_font_size=10)
        fig.update_layout(**CHART_LAYOUT, height=220, showlegend=False,
                          yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ─── SECTION 5: COMPARATIVE ANALYSIS ─────────────────────────────────────────
st.markdown('<div class="section-heading">Comparative Analysis</div>', unsafe_allow_html=True)
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

col_x, col_y = st.columns(2, gap="large")

with col_x:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Healthy vs At-Risk — Sensor Averages</div>', unsafe_allow_html=True)
    avg_df   = df2.groupby("Status")[FEATURES].mean().reset_index()
    avg_melt = avg_df.melt(id_vars="Status", var_name="Feature", value_name="Value")
    avg_melt["Feature"] = avg_melt["Feature"].map(dict(zip(FEATURES, FEAT_LABELS)))
    for f in FEAT_LABELS:
        mx = avg_melt.loc[avg_melt["Feature"]==f,"Value"].max()
        mn = avg_melt.loc[avg_melt["Feature"]==f,"Value"].min()
        if mx != mn:
            avg_melt.loc[avg_melt["Feature"]==f,"Value"] = \
                (avg_melt.loc[avg_melt["Feature"]==f,"Value"]-mn)/(mx-mn)
    fig = px.bar(avg_melt, x="Feature", y="Value", color="Status",
                 barmode="group",
                 color_discrete_map={"Healthy":"#4a7c59","At Risk":"#e57373"},
                 labels={"Value":"Normalised Score (0–1)","Feature":""},
                 height=260)
    fig.update_layout(**CHART_LAYOUT, legend=dict(orientation="h",y=-0.2),
                      xaxis_tickangle=-20,
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box insight-blue">
      <div class="insight-title">📖 How to read this</div>
      Bars closer to 1.0 are higher for that group. Taller green bars = healthy plants 
      have better readings in that sensor. If red bars are taller, at-risk plants have 
      abnormally high values for that feature.
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_y:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📦 Soil pH Range by Health Status</div>', unsafe_allow_html=True)
    fig = px.box(df2, x="Status", y="Soil_pH",
                 color="Status",
                 color_discrete_map={"Healthy":"#4a7c59","At Risk":"#e57373"},
                 points="outliers",
                 labels={"Soil_pH":"Soil pH","Status":""},
                 height=260)
    fig.add_hrect(y0=5.5, y1=7.0, fillcolor="rgba(74,124,89,0.08)",
                  line_width=1, line_color="#4a7c59",
                  annotation_text="Optimal pH 5.5–7.0",
                  annotation_font_size=10, annotation_font_color="#4a7c59")
    fig.update_layout(**CHART_LAYOUT, showlegend=False,
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    healthy_ph  = df2[df2["Status"]=="Healthy"]["Soil_pH"].median()
    atrisk_ph   = df2[df2["Status"]=="At Risk"]["Soil_pH"].median()
    st.markdown(f"""
    <div class="insight-box insight-amber">
      <div class="insight-title">🔬 pH Insight</div>
      Healthy plants have a median pH of <b>{healthy_ph:.1f}</b>, 
      while at-risk plants sit at <b>{atrisk_ph:.1f}</b>. 
      The green shaded band shows the ideal range (5.5–7.0).
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Scatter + Nutrient bar
col_s1, col_s2 = st.columns([1.4, 1], gap="large")

with col_s1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌡 Temperature vs Humidity · coloured by Health</div>', unsafe_allow_html=True)
    sdf = df2.sample(min(600, len(df2)), random_state=1)
    fig = px.scatter(sdf, x="Temperature_C", y="Humidity_%",
                     color="Status",
                     color_discrete_map={"Healthy":"#4a7c59","At Risk":"#e57373"},
                     opacity=0.6, size_max=7,
                     labels={"Temperature_C":"Temperature (°C)","Humidity_%":"Humidity (%)"},
                     height=280)
    fig.add_vrect(x0=18, x1=28, fillcolor="rgba(74,124,89,0.07)", line_width=0)
    fig.add_hrect(y0=50, y1=80, fillcolor="rgba(74,124,89,0.07)", line_width=0)
    fig.update_layout(**CHART_LAYOUT, legend=dict(orientation="h",y=-0.2),
                      xaxis=dict(showgrid=True, gridcolor="#e8f5ee"),
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="insight-box insight-blue">
      <div class="insight-title">🗺️ Reading the scatter</div>
      Green dots in the shaded box (ideal temp & humidity) tend to be healthy. 
      Red dots outside this zone signal environmental stress.
    </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_s2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧪 Average Nutrient Level by Health</div>', unsafe_allow_html=True)
    nutri_avg = df2.groupby("Status")["Nutrient_Level"].mean().reset_index()
    fig = px.bar(nutri_avg, x="Status", y="Nutrient_Level",
                 color="Status",
                 color_discrete_map={"Healthy":"#4a7c59","At Risk":"#e57373"},
                 labels={"Nutrient_Level":"Avg Nutrient Level","Status":""},
                 text_auto=".1f", height=200)
    fig.update_traces(textposition="outside")
    fig.update_layout(**CHART_LAYOUT, showlegend=False,
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    h_n = nutri_avg[nutri_avg["Status"]=="Healthy"]["Nutrient_Level"].values
    r_n = nutri_avg[nutri_avg["Status"]=="At Risk"]["Nutrient_Level"].values
    if len(h_n) and len(r_n):
        diff = h_n[0] - r_n[0]
        st.markdown(f"""
        <div class="insight-box {'insight-green' if diff>0 else 'insight-red'}">
          <div class="insight-title">💡 Key finding</div>
          Healthy plants average <b>{h_n[0]:.1f}</b> nutrient level vs 
          <b>{r_n[0]:.1f}</b> for at-risk plants — 
          a difference of <b>{abs(diff):.1f} points</b>.
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── SECTION 6: AI PREDICTION ─────────────────────────────────────────────────
st.markdown('<div class="section-heading">🤖 Live AI Prediction</div>', unsafe_allow_html=True)
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("Adjust the sliders to simulate your plant's sensor readings, then click **Predict Plant Health** for an instant AI assessment.")

sc1, sc2, sc3 = st.columns(3)
with sc1:
    temp     = st.slider("🌡 Temperature (°C)",    10.0, 45.0, 25.0, 0.5)
    humidity = st.slider("💧 Humidity (%)",         20.0, 100.0, 60.0, 1.0)
with sc2:
    moisture = st.slider("🪴 Soil Moisture (%)",    5.0, 100.0, 40.0, 1.0)
    ph       = st.slider("⚗️ Soil pH",              3.0, 10.0,  6.5, 0.1)
with sc3:
    nutrient = st.slider("🧪 Nutrient Level",       10.0, 100.0, 50.0, 1.0)
    light    = st.slider("☀️ Light Intensity (lux)", 1000.0, 30000.0, 15000.0, 500.0)

# Real-time range check badges (no button needed)
st.markdown("**Quick Check — Is each reading in the optimal range?**")
badge_cols = st.columns(6)
slider_vals = {
    "Temperature_C": temp,
    "Humidity_%": humidity,
    "Soil_Moisture_%": moisture,
    "Soil_pH": ph,
    "Nutrient_Level": nutrient,
    "Light_Intensity_lux": light,
}
for i, (feat, lbl) in enumerate(zip(FEATURES, FEAT_LABELS)):
    lo, hi, unit, _ = OPTIMAL[feat]
    val = slider_vals[feat]
    in_ok = lo <= val <= hi
    with badge_cols[i]:
        color = "#d1fae5" if in_ok else "#fee2e2"
        txt_color = "#065f46" if in_ok else "#991b1b"
        icon = "✅" if in_ok else "⚠️"
        st.markdown(f"""
        <div style="background:{color};border-radius:12px;padding:10px 12px;text-align:center;margin-bottom:8px;">
          <div style="font-size:1.3rem;">{icon}</div>
          <div style="font-size:0.72rem;font-weight:700;color:{txt_color};margin-top:2px;">{lbl}</div>
          <div style="font-size:0.7rem;color:{txt_color};">{val:.1f}{unit}</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
pred_col, radar_col = st.columns([1, 1.8], gap="large")

with pred_col:
    if st.button("🌱 Predict Plant Health"):
        inp = np.array([[temp, humidity, moisture, ph, nutrient, light]])

        # ── FIXED THRESHOLD LOGIC ────────────────────────────────────────────
        prob           = model.predict_proba(inp)[0]
        classes        = list(model.classes_)
        unhealthy_prob = prob[classes.index(0)]
        healthy_prob   = prob[classes.index(1)]

        THRESHOLD = 0.65   # tune between 0.60–0.70 as needed
        pred = 1 if healthy_prob >= THRESHOLD else 0
        # ─────────────────────────────────────────────────────────────────────

        if pred == 1:
            st.markdown(f"""
            <div style="background:#d1fae5;border-radius:16px;padding:24px;text-align:center;margin-top:12px;">
              <div style="font-size:3rem">🌿</div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#065f46;margin:8px 0">Healthy Plant</div>
              <div style="font-size:0.9rem;color:#047857">Confidence: {healthy_prob*100:.1f}%</div>
              <div style="font-size:0.8rem;color:#6b8f71;margin-top:6px;">Keep up current conditions!</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background:#fee2e2;border-radius:16px;padding:24px;text-align:center;margin-top:12px;">
              <div style="font-size:3rem">⚠️</div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#991b1b;margin:8px 0">Health Risk Detected</div>
              <div style="font-size:0.9rem;color:#b91c1c">Confidence: {unhealthy_prob*100:.1f}%</div>
              <div style="font-size:0.8rem;color:#6b8f71;margin-top:6px;">Review highlighted sensors above.</div>
            </div>""", unsafe_allow_html=True)

        # Probability gradient bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div>
          <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#6b8f71;margin-bottom:4px;">
            <span>At Risk</span><span>Healthy</span>
          </div>
          <div style="height:10px;border-radius:99px;background:#fee2e2;overflow:hidden;">
            <div style="width:{healthy_prob*100:.1f}%;height:100%;background:linear-gradient(90deg,#e57373,#4a7c59);border-radius:99px;"></div>
          </div>
          <div style="text-align:center;font-size:0.8rem;color:#6b8f71;margin-top:4px;">
            {healthy_prob*100:.1f}% healthy · threshold set at {THRESHOLD*100:.0f}%
          </div>
        </div>""", unsafe_allow_html=True)

        # Plain-English recommendations
        recs = []
        if not (18 <= temp <= 28):
            recs.append(f"🌡 Temperature ({temp}°C) is outside 18–28°C ideal — {'cool it down' if temp>28 else 'add warmth'}.")
        if not (50 <= humidity <= 80):
            recs.append(f"💧 Humidity ({humidity}%) should be 50–80% — {'reduce moisture' if humidity>80 else 'mist or humidify'}.")
        if not (30 <= moisture <= 70):
            recs.append(f"🪴 Soil moisture ({moisture}%) is {'too wet — check drainage' if moisture>70 else 'too dry — water now'}.")
        if not (5.5 <= ph <= 7.0):
            recs.append(f"⚗️ Soil pH ({ph}) is outside 5.5–7.0 — {'add lime to raise' if ph<5.5 else 'add sulfur to lower'} pH.")
        if nutrient < 50:
            recs.append(f"🧪 Nutrient level ({nutrient:.0f}) is low — consider fertilising.")
        if not (8000 <= light <= 25000):
            recs.append(f"☀️ Light ({light/1000:.0f}k lux) is {'too intense — add shade' if light>25000 else 'too low — move to brighter spot'}.")

        if recs:
            st.markdown("<br>", unsafe_allow_html=True)
            rec_html = "".join([f"<li style='margin-bottom:6px;'>{r}</li>" for r in recs])
            st.markdown(f"""
            <div class="insight-box insight-amber">
              <div class="insight-title">🛠️ What to fix</div>
              <ul style="margin:8px 0 0 16px;padding:0;">{rec_html}</ul>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="insight-box insight-green" style="margin-top:12px;">
              <div class="insight-title">🎉 All conditions look great!</div>
              Every reading is within the optimal range. Continue monitoring regularly.
            </div>""", unsafe_allow_html=True)

with radar_col:
    feat_vals = [temp, humidity, moisture, ph, nutrient, light/1000]
    feat_avgs = [df[f].mean() for f in FEATURES[:-1]] + [df["Light_Intensity_lux"].mean()/1000]
    radar_labels = ["Temp","Humidity","Moisture","pH","Nutrient","Light (klux)"]
    maxes = [45, 100, 100, 10, 100, 30]
    norm_vals = [v/m for v,m in zip(feat_vals, maxes)]
    norm_avgs = [v/m for v,m in zip(feat_avgs, maxes)]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=norm_avgs + [norm_avgs[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        fillcolor="rgba(74,124,89,0.15)",
        line=dict(color="#4a7c59", width=2),
        name="Dataset Average"
    ))
    fig.add_trace(go.Scatterpolar(
        r=norm_vals + [norm_vals[0]],
        theta=radar_labels + [radar_labels[0]],
        fill="toself",
        fillcolor="rgba(244,162,97,0.18)",
        line=dict(color="#f4a261", width=2.5, dash="dot"),
        name="Your Input"
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0,1], showticklabels=False, gridcolor="#e8f5ee"),
            angularaxis=dict(gridcolor="#e8f5ee"),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=True,
        legend=dict(orientation="h", y=-0.12),
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="DM Sans",
        font_color="#1b3a2d",
        title=dict(text="Your Input vs Dataset Average", font_family="DM Serif Display",
                   font_size=14, font_color="#2d6a4f"),
        height=340,
        margin=dict(l=40,r=40,t=50,b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-footer">
  🌱 PlantPulse AI · Powered by Random Forest · Built with Streamlit & Plotly<br>
  Threshold: {0.65*100:.0f}% · Dataset: {data_source} · {total:,} records
</div>
""", unsafe_allow_html=True)