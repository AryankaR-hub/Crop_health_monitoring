import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="🌱 PlantPulse AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─── GLOBAL CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── ROOT PALETTE ── */
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

/* ── BASE ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}

.stApp {
    background: linear-gradient(160deg, #f0f7f2 0%, #e8f5ee 40%, #f5f0e8 100%);
    min-height: 100vh;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 3rem; max-width: 1400px; }

/* ── HEADER ── */
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
.dash-header::after {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: rgba(255,255,255,0.07);
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

/* ── CARDS ── */
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

/* ── KPI TILES ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
    margin-bottom: 24px;
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

/* ── STATUS BADGE ── */
.status-healthy {
    display: inline-flex; align-items: center; gap: 6px;
    background: #d1fae5; color: #065f46;
    border-radius: 999px; padding: 6px 16px;
    font-weight: 600; font-size: 0.85rem;
}
.status-risk {
    display: inline-flex; align-items: center; gap: 6px;
    background: #fee2e2; color: #991b1b;
    border-radius: 999px; padding: 6px 16px;
    font-weight: 600; font-size: 0.85rem;
}
.pred-dot-healthy { width:9px; height:9px; border-radius:50%; background:#10b981; }
.pred-dot-risk    { width:9px; height:9px; border-radius:50%; background:#ef4444; }

/* ── SECTION HEADING ── */
.section-heading {
    font-family: 'DM Serif Display', serif;
    font-size: 1.45rem;
    color: var(--leaf);
    margin: 32px 0 16px;
    border-left: 4px solid var(--sprout);
    padding-left: 14px;
}

/* ── PROGRESS BARS ── */
.prog-wrap { margin-bottom: 14px; }
.prog-label {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; color: var(--muted); margin-bottom: 5px;
}
.prog-bar-bg {
    height: 8px; border-radius: 99px;
    background: var(--foam);
    overflow: hidden;
}
.prog-bar-fill {
    height: 100%; border-radius: 99px;
    transition: width 0.6s ease;
}

/* ── PREDICT BUTTON ── */
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

/* ── SLIDERS ── */
[data-baseweb="slider"] [role="slider"] {
    background: var(--sage) !important;
    border-color: var(--sage) !important;
}
[data-baseweb="slider"] [data-testid="stTickBar"] { color: var(--muted); }

/* ── DIVIDER ── */
.green-divider {
    height: 2px;
    background: linear-gradient(90deg, var(--sprout), transparent);
    border: none;
    margin: 8px 0 24px;
    border-radius: 2px;
}

/* ── FOOTER ── */
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
MULTI_SEQ = ["#4a7c59","#90caf9","#f4a261","#e57373","#a78bfa","#f472b6"]

# ─── DATA & MODEL ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    d = pd.read_csv("plant_moniter_health_data.csv").dropna()
    d["Soil_Moisture_%"] = d["Soil_Moisture_%"].abs()
    d.index = pd.RangeIndex(len(d))
    return d

@st.cache_resource
def load_model():
    return joblib.load("random_forest_model.pkl")

df    = load_data()
model = load_model()

# ─── FEATURE IMPORTANCE ────────────────────────────────────────────────────────
FEATURES = ["Temperature_C","Humidity_%","Soil_Moisture_%","Soil_pH","Nutrient_Level","Light_Intensity_lux"]
FEAT_LABELS = ["Temperature","Humidity","Soil Moisture","Soil pH","Nutrient Level","Light Intensity"]
importance = model.feature_importances_

# ─── HEALTH STATUS ─────────────────────────────────────────────────────────────
health_counts = df["Health_Status"].value_counts()
total = len(df)
healthy_pct   = round(health_counts.get(1, health_counts.get("Healthy", 0)) / total * 100, 1)

# ─── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dash-header">
  <h1>🌱 PlantPulse AI</h1>
  <p>Smart Plant Health Monitoring · {total:,} records · Random Forest Model Active</p>
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

# Donut — Health Status Distribution
with col_a:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌿 Health Status Distribution</div>', unsafe_allow_html=True)
    hc = df["Health_Status"].value_counts().reset_index()
    hc.columns = ["Status","Count"]
    hc["Label"] = hc["Status"].map({1:"Healthy", 0:"At Risk", "Healthy":"Healthy", "Unhealthy":"At Risk"}).fillna(hc["Status"].astype(str))
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
    st.markdown('</div>', unsafe_allow_html=True)

# Donut — Nutrient Level buckets
with col_b:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧪 Nutrient Level Spread</div>', unsafe_allow_html=True)
    bins   = pd.cut(df["Nutrient_Level"], bins=[0,30,60,80,101], labels=["Low","Medium","Good","High"])
    ndf    = bins.value_counts().sort_index().reset_index()
    ndf.columns = ["Range","Count"]
    fig = go.Figure(go.Pie(
        labels=ndf["Range"], values=ndf["Count"],
        hole=0.55,
        marker_colors=["#e57373","#f4a261","#74c69d","#2d6a4f"],
        textinfo="percent+label",
        textfont_size=11,
        hovertemplate="%{label}: %{value}<extra></extra>"
    ))
    fig.update_layout(**CHART_LAYOUT, showlegend=False, height=280)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Feature importance bar
with col_c:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🤖 Model Feature Importance</div>', unsafe_allow_html=True)
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
                      yaxis_title="", height=280,
                      xaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── SECTION 2: ENVIRONMENT TRENDS ────────────────────────────────────────────
st.markdown('<div class="section-heading">Environment Trends</div>', unsafe_allow_html=True)
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

# Rolling averages (smooth)
window = max(5, len(df)//100)
df_roll = df[FEATURES].rolling(window, min_periods=1).mean()
df_roll.index = df.index

col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌡 Temperature & Humidity (Smoothed)</div>', unsafe_allow_html=True)
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
    fig.update_layout(**CHART_LAYOUT, height=260,
                      legend=dict(orientation="h", y=-0.15),
                      xaxis=dict(showgrid=False, title="Record Index"),
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_r:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🪴 Soil Moisture & pH (Smoothed)</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df_roll["Soil_Moisture_%"], name="Soil Moisture %",
        line=dict(color="#74c69d", width=2.5),
        fill="tozeroy", fillcolor="rgba(116,198,157,0.12)",
        hovertemplate="Moisture: %{y:.1f}%<extra></extra>"
    ))
    fig.add_trace(go.Scatter(
        y=df_roll["Soil_pH"]*10, name="Soil pH (×10)",
        line=dict(color="#f4a261", width=2.5, dash="dot"),
        hovertemplate="pH: %{customdata:.1f}<extra></extra>",
        customdata=df_roll["Soil_pH"]
    ))
    fig.update_layout(**CHART_LAYOUT, height=260,
                      legend=dict(orientation="h", y=-0.15),
                      xaxis=dict(showgrid=False, title="Record Index"),
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── SECTION 3: DISTRIBUTIONS ─────────────────────────────────────────────────
st.markdown('<div class="section-heading">Feature Distributions</div>', unsafe_allow_html=True)
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3, gap="large")

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌡 Temperature Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(df, x="Temperature_C", nbins=30,
                       color_discrete_sequence=["#e57373"],
                       labels={"Temperature_C":"Temperature (°C)"})
    fig.update_traces(marker_line_width=0.5, marker_line_color="white", opacity=0.85)
    fig.update_layout(**CHART_LAYOUT, height=230, showlegend=False,
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">💧 Soil Moisture Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(df, x="Soil_Moisture_%", nbins=30,
                       color_discrete_sequence=["#74c69d"],
                       labels={"Soil_Moisture_%":"Soil Moisture (%)"})
    fig.update_traces(marker_line_width=0.5, marker_line_color="white", opacity=0.85)
    fig.update_layout(**CHART_LAYOUT, height=230, showlegend=False,
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">☀️ Light Intensity Distribution</div>', unsafe_allow_html=True)
    fig = px.histogram(df, x="Light_Intensity_lux", nbins=30,
                       color_discrete_sequence=["#f4a261"],
                       labels={"Light_Intensity_lux":"Light (lux)"})
    fig.update_traces(marker_line_width=0.5, marker_line_color="white", opacity=0.85)
    fig.update_layout(**CHART_LAYOUT, height=230, showlegend=False,
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── SECTION 4: CORRELATION & BOX PLOTS ───────────────────────────────────────
st.markdown('<div class="section-heading">Comparative Analysis</div>', unsafe_allow_html=True)
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

col_x, col_y = st.columns(2, gap="large")

with col_x:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📊 Feature Averages by Health Status</div>', unsafe_allow_html=True)
    status_map = {1:"Healthy", 0:"At Risk", "Healthy":"Healthy", "Unhealthy":"At Risk"}
    df2 = df.copy()
    df2["Status"] = df2["Health_Status"].map(status_map).fillna(df2["Health_Status"].astype(str))
    avg_df = df2.groupby("Status")[FEATURES].mean().reset_index()
    avg_melt = avg_df.melt(id_vars="Status", var_name="Feature", value_name="Value")
    avg_melt["Feature"] = avg_melt["Feature"].map(dict(zip(FEATURES, FEAT_LABELS)))
    # Normalize each feature 0-1 for easy visual comparison
    for f in FEAT_LABELS:
        mx = avg_melt.loc[avg_melt["Feature"]==f,"Value"].max()
        mn = avg_melt.loc[avg_melt["Feature"]==f,"Value"].min()
        if mx != mn:
            avg_melt.loc[avg_melt["Feature"]==f,"Value"] = (avg_melt.loc[avg_melt["Feature"]==f,"Value"]-mn)/(mx-mn)
    fig = px.bar(avg_melt, x="Feature", y="Value", color="Status",
                 barmode="group",
                 color_discrete_map={"Healthy":"#4a7c59","At Risk":"#e57373"},
                 labels={"Value":"Normalised Value","Feature":""},
                 height=280)
    fig.update_layout(**CHART_LAYOUT,
                      legend=dict(orientation="h",y=-0.2),
                      xaxis_tickangle=-20,
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_y:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">📦 Soil pH by Health Status</div>', unsafe_allow_html=True)
    df2["Status"] = df2["Health_Status"].map(status_map).fillna(df2["Health_Status"].astype(str))
    fig = px.box(df2, x="Status", y="Soil_pH",
                 color="Status",
                 color_discrete_map={"Healthy":"#4a7c59","At Risk":"#e57373"},
                 points="outliers",
                 labels={"Soil_pH":"Soil pH","Status":""},
                 height=280)
    fig.update_layout(**CHART_LAYOUT, showlegend=False,
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── SECTION 5: SCATTER & NUTRIENT BAR ────────────────────────────────────────
col_s1, col_s2 = st.columns([1.4, 1], gap="large")

with col_s1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🌡 Temperature vs Humidity · coloured by Health</div>', unsafe_allow_html=True)
    sdf = df2.sample(min(500, len(df2)), random_state=1)
    fig = px.scatter(sdf, x="Temperature_C", y="Humidity_%",
                     color="Status",
                     color_discrete_map={"Healthy":"#4a7c59","At Risk":"#e57373"},
                     opacity=0.65, size_max=7,
                     labels={"Temperature_C":"Temperature (°C)","Humidity_%":"Humidity (%)"},
                     height=280)
    fig.update_layout(**CHART_LAYOUT, legend=dict(orientation="h",y=-0.2),
                      xaxis=dict(showgrid=True, gridcolor="#e8f5ee"),
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_s2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="card-title">🧪 Avg Nutrient by Health Status</div>', unsafe_allow_html=True)
    nutri_avg = df2.groupby("Status")["Nutrient_Level"].mean().reset_index()
    fig = px.bar(nutri_avg, x="Status", y="Nutrient_Level",
                 color="Status",
                 color_discrete_map={"Healthy":"#4a7c59","At Risk":"#e57373"},
                 labels={"Nutrient_Level":"Avg Nutrient Level","Status":""},
                 text_auto=".1f", height=280)
    fig.update_traces(textposition="outside")
    fig.update_layout(**CHART_LAYOUT, showlegend=False,
                      yaxis=dict(showgrid=True, gridcolor="#e8f5ee"))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─── SECTION 6: AI PREDICTION ─────────────────────────────────────────────────
st.markdown('<div class="section-heading">🤖 Live AI Prediction</div>', unsafe_allow_html=True)
st.markdown('<hr class="green-divider">', unsafe_allow_html=True)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("Adjust the sliders below to simulate sensor readings and get an instant health prediction from the Random Forest model.")

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

# Radar chart of user input vs dataset averages
feat_vals = [temp, humidity, moisture, ph, nutrient, light/1000]
feat_avgs = [df["Temperature_C"].mean(), df["Humidity_%"].mean(),
             df["Soil_Moisture_%"].mean(), df["Soil_pH"].mean(),
             df["Nutrient_Level"].mean(), df["Light_Intensity_lux"].mean()/1000]
radar_labels = ["Temp","Humidity","Moisture","pH","Nutrient","Light (klux)"]

# Normalise 0-1 for radar
maxes = [45, 100, 100, 10, 100, 30]
norm_vals = [v/m for v,m in zip(feat_vals, maxes)]
norm_avgs = [v/m for v,m in zip(feat_avgs, maxes)]

st.markdown("<br>", unsafe_allow_html=True)
pred_col, radar_col = st.columns([1, 1.8], gap="large")

with pred_col:
    if st.button("🌱 Predict Plant Health"):
        inp  = np.array([[temp, humidity, moisture, ph, nutrient, light]])
        pred = model.predict(inp)[0]
        prob = model.predict_proba(inp)[0]
        healthy_prob = prob[list(model.classes_).index(1)] if 1 in model.classes_ else prob.max()

        if pred == 1 or str(pred).lower() in ["healthy","1"]:
            st.markdown(f"""
            <div style="background:#d1fae5;border-radius:16px;padding:24px;text-align:center;margin-top:12px;">
              <div style="font-size:3rem">🌿</div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#065f46;margin:8px 0">Healthy Plant</div>
              <div style="font-size:0.9rem;color:#047857">Confidence: {healthy_prob*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        else:
            at_risk_prob = 1 - healthy_prob
            st.markdown(f"""
            <div style="background:#fee2e2;border-radius:16px;padding:24px;text-align:center;margin-top:12px;">
              <div style="font-size:3rem">⚠️</div>
              <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;color:#991b1b;margin:8px 0">Health Risk Detected</div>
              <div style="font-size:0.9rem;color:#b91c1c">Confidence: {at_risk_prob*100:.1f}%</div>
            </div>""", unsafe_allow_html=True)

        # Probability bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="margin-top:8px">
          <div style="display:flex;justify-content:space-between;font-size:0.8rem;color:#6b8f71;margin-bottom:4px">
            <span>At Risk</span><span>Healthy</span>
          </div>
          <div style="height:10px;border-radius:99px;background:#fee2e2;overflow:hidden">
            <div style="width:{healthy_prob*100:.1f}%;height:100%;background:linear-gradient(90deg,#e57373,#4a7c59);border-radius:99px"></div>
          </div>
          <div style="text-align:center;font-size:0.8rem;color:#6b8f71;margin-top:4px">{healthy_prob*100:.1f}% healthy probability</div>
        </div>""", unsafe_allow_html=True)

with radar_col:
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
        height=320,
        margin=dict(l=40,r=40,t=50,b=40)
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ─── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-footer">
  🌱 PlantPulse AI · Powered by Random Forest · Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)