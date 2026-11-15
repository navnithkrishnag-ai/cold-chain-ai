"""
ColdGuard AI - Main Application & Executive Overview Dashboard
"""

import streamlit as st
import pandas as pd
import config
from src.ui_components import (
    inject_custom_css,
    initialize_session_state,
    render_sidebar,
    apply_plotly_theme
)
from src.risk_assessment import assess_cold_chain_risk
from src.alert_system import generate_cold_chain_alerts
import plotly.express as px
import plotly.graph_objects as go

# Streamlit Page Setup
st.set_page_config(
    page_title=f"{config.APP_NAME} - Overview",
    page_icon=config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded"
)

inject_custom_css()
initialize_session_state()
render_sidebar()

# Page Header
st.title(f"{config.APP_ICON} {config.APP_NAME}: System Overview")
st.markdown(
    "**AI-Enabled Cold-Chain Condition Monitoring & Early Warning System** — "
    "Unsupervised ML anomaly detection and multi-factor thermal risk evaluation."
)

df = st.session_state.df

if df.empty:
    st.warning("⚠️ No dataset loaded. Please go to **Data Upload** in the sidebar to upload records or load a sample dataset.")
    st.stop()

# Recompute risk metrics based on current sidebar thresholds
risk_eval = assess_cold_chain_risk(
    df,
    min_temp=st.session_state.min_temp,
    max_temp=st.session_state.max_temp,
    max_allowed_excursion_minutes=st.session_state.max_excursion_min
)
metrics = risk_eval.get("metrics", {})
category = risk_eval.get("risk_category", "UNKNOWN")
score = risk_eval.get("risk_score", 0)

# Execution mode styling
mode = st.session_state.execution_mode
mode_badge = (
    '<span class="badge-local">LOCAL DEMO MODE (Scikit-Learn CPU)</span>'
    if mode == "LOCAL" else
    '<span class="badge-aws">AWS CLOUD MODE (Amazon SageMaker AI)</span>'
)

# Status Callout
badge_class = f"badge-risk-{category.lower()}"
st.markdown(
    f"""
    <div style="background-color: #12253B; border-left: 5px solid #38BDF8; padding: 14px 20px; border-radius: 8px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-size: 1.1rem; font-weight: 700; color: #F8FAFC;">Cold-Chain Status: </span>
                <span class="{badge_class}">{category} RISK (Score: {score}/100)</span>
            </div>
            <div style="margin-top: 5px;">
                <span style="font-size: 0.85rem; color: #94A3B8; margin-right: 8px;">Active Engine:</span>
                {mode_badge}
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ==========================================
# PAGE 1 REQUIRED KPIS
# ==========================================
col1, col2, col3, col4, col5, col6 = st.columns(6)

total_sensors = df["sensor_id"].nunique() if "sensor_id" in df else 1
total_obs = len(df)
avg_temp = df["temperature"].mean()
avg_hum = df["humidity"].mean() if "humidity" in df else 0.0
total_anomalies = int(df["is_anomaly"].sum()) if "is_anomaly" in df else 0
high_risk_records = int((df["temperature"] > st.session_state.max_temp).sum() + (df["temperature"] < st.session_state.min_temp).sum())

with col1:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Total Sensors</div>
            <div class="metric-value">{total_sensors}</div>
            <div class="metric-sub">Active DHT22 Loggers</div>
        </div>""",
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Observations</div>
            <div class="metric-value">{total_obs:,}</div>
            <div class="metric-sub">Chronological Steps</div>
        </div>""",
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Average Temp</div>
            <div class="metric-value">{avg_temp:.1f}°C</div>
            <div class="metric-sub">Safe: {st.session_state.min_temp}°C to {st.session_state.max_temp}°C</div>
        </div>""",
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Average Humidity</div>
            <div class="metric-value">{avg_hum:.1f}%</div>
            <div class="metric-sub">Relative Humidity (RH)</div>
        </div>""",
        unsafe_allow_html=True
    )

with col5:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">ML Anomalies</div>
            <div class="metric-value">{total_anomalies}</div>
            <div class="metric-sub">{metrics.get('anomaly_frequency_pct', 0.0)}% of total readings</div>
        </div>""",
        unsafe_allow_html=True
    )

with col6:
    st.markdown(
        f"""<div class="metric-card">
            <div class="metric-label">Excursions</div>
            <div class="metric-value">{high_risk_records}</div>
            <div class="metric-sub">{metrics.get('percent_out_of_range', 0.0)}% out of nominal range</div>
        </div>""",
        unsafe_allow_html=True
    )

st.markdown("---")

# Visual Overview Section
row_col1, row_col2 = st.columns([7, 5])

with row_col1:
    st.subheader("📊 Live Cold-Storage Temperature Trend")
    # Temperature chart with acceptable threshold lines
    fig_temp = px.line(
        df,
        x="timestamp",
        y="temperature",
        color="sensor_id",
        labels={"timestamp": "Time", "temperature": "Temperature (°C)", "sensor_id": "Sensor"}
    )
    # Add acceptable range horizontal lines
    fig_temp.add_hline(
        y=st.session_state.max_temp,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Max Safe Limit ({st.session_state.max_temp}°C)",
        annotation_position="top right"
    )
    fig_temp.add_hline(
        y=st.session_state.min_temp,
        line_dash="dash",
        line_color="#38BDF8",
        annotation_text=f"Min Safe Limit ({st.session_state.min_temp}°C)",
        annotation_position="bottom right"
    )
    fig_temp = apply_plotly_theme(fig_temp, "Multi-Sensor Temperature Trajectory vs Safe Operating Band")
    st.plotly_chart(fig_temp, use_container_width=True)

with row_col2:
    st.subheader("🛡️ Risk Score Factor Contribution")
    breakdown = risk_eval.get("point_breakdown", {})
    breakdown_df = pd.DataFrame([
        {"Factor": "Excursion Duration", "Points": breakdown.get("duration_points", 0)},
        {"Factor": "Peak Severity", "Points": breakdown.get("severity_points", 0)},
        {"Factor": "ML Anomaly Rate", "Points": breakdown.get("ml_anomaly_points", 0)},
        {"Factor": "Thermal Trend", "Points": breakdown.get("trend_points", 0)}
    ])
    fig_risk = px.bar(
        breakdown_df,
        x="Points",
        y="Factor",
        orientation="h",
        color="Points",
        color_continuous_scale=["#10B981", "#F59E0B", "#EF4444"],
        range_color=[0, 40]
    )
    fig_risk = apply_plotly_theme(fig_risk, f"Points Breakdown (Total Score: {score}/100)")
    fig_risk.update_layout(coloraxis_showscale=False)
    st.plotly_chart(fig_risk, use_container_width=True)

# Alerts Feed Snapshot
st.subheader("🚨 Recent Operational Alerts")
alerts_df = generate_cold_chain_alerts(
    df,
    min_temp=st.session_state.min_temp,
    max_temp=st.session_state.max_temp
)

if alerts_df.empty:
    st.success("✅ No active operational alerts detected. All readings conform to standard conditions.")
else:
    # Display recent 5 alerts
    st.dataframe(
        alerts_df[["timestamp", "sensor_id", "severity", "alert_type", "temperature", "anomaly_score", "reason"]].tail(8),
        use_container_width=True,
        hide_index=True
    )

st.info(
    "💡 **Cold-Chain Operational Notice**: ColdGuard AI is designed to assist warehouse managers and operators with "
    "early condition warnings. It should not be used as an absolute guarantee against mechanical failure or spoilage."
)
