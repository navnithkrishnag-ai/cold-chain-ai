"""
ColdGuard AI - PAGE 5: COLD-CHAIN RISK ASSESSMENT & ALERTS
Evaluates rule-based thermal integrity, duration of excursions, ML anomaly frequency,
and transparent classification into LOW, MODERATE, HIGH, or CRITICAL risk.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import config
from src.ui_components import inject_custom_css, initialize_session_state, render_sidebar, apply_plotly_theme
from src.risk_assessment import assess_cold_chain_risk
from src.alert_system import generate_cold_chain_alerts

st.set_page_config(
    page_title=f"{config.APP_NAME} - Risk Assessment",
    page_icon="🛡️",
    layout="wide"
)

inject_custom_css()
initialize_session_state()
render_sidebar()

st.title("🛡️ Cold-Chain Risk Assessment Engine")
st.markdown(
    "Multi-factor rule-based risk evaluation combining thermal boundaries, excursion durations, "
    "and ML anomaly frequencies. Provides transparent factor attribution and actionable operational recommendations."
)

st.caption(
    "ℹ️ **System Disclaimer**: ColdGuard AI is a condition monitoring and early warning system designed to assist cold-chain operators. "
    "It does not claim to predict mechanical failures or product spoilage with certainty."
)

df = st.session_state.df

if df.empty:
    st.warning("⚠️ No dataset loaded. Please go to **Data Upload** to load a dataset.")
    st.stop()

# Run Risk Engine
risk_data = assess_cold_chain_risk(
    df,
    min_temp=st.session_state.min_temp,
    max_temp=st.session_state.max_temp,
    max_allowed_excursion_minutes=st.session_state.max_excursion_min
)

category = risk_data.get("risk_category", "UNKNOWN")
score = risk_data.get("risk_score", 0)
metrics = risk_data.get("metrics", {})
breakdown = risk_data.get("point_breakdown", {})
reasons = risk_data.get("reasons", [])
recommendations = risk_data.get("recommendations", [])

# ==========================================
# 1. RISK CATEGORY & SCORE BANNER
# ==========================================
badge_color_map = {
    "LOW": ("#064E3B", "#6EE7B7"),
    "MODERATE": ("#78350F", "#FDE68A"),
    "HIGH": ("#7C2D12", "#FDBA74"),
    "CRITICAL": ("#7F1D1D", "#FCA5A5")
}
bg_c, text_c = badge_color_map.get(category, ("#1E3E62", "#38BDF8"))

st.markdown(
    f"""
    <div style="background-color: {bg_c}; border-radius: 10px; padding: 22px 28px; margin-bottom: 25px; border: 1px solid {text_c};">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span style="font-size: 0.95rem; text-transform: uppercase; letter-spacing: 0.05em; color: {text_c}; font-weight: 600;">Overall Cold-Chain Risk Level</span>
                <h1 style="margin: 4px 0 0 0; font-size: 2.4rem; color: #FFFFFF; font-weight: 800;">{category} RISK</h1>
            </div>
            <div style="text-align: right;">
                <div style="font-size: 0.9rem; color: {text_c};">Risk Index Score</div>
                <div style="font-size: 2.8rem; font-weight: 800; color: #FFFFFF;">{score} <span style="font-size: 1.2rem; color: {text_c};">/ 100</span></div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Metrics Cards
mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
mcol1.metric("Est. Excursion Time", f"{metrics.get('excursion_duration_minutes', 0.0)} min", f"Max: {metrics.get('max_allowed_excursion_minutes')} min")
mcol2.metric("Excursion Episodes", f"{metrics.get('num_excursion_episodes', 0)} events")
mcol3.metric("Peak Temperature", f"{metrics.get('max_recorded_temp', 'N/A')}°C", f"Limit: {metrics.get('max_configured_temp')}°C")
mcol4.metric("Lowest Temperature", f"{metrics.get('min_recorded_temp', 'N/A')}°C", f"Limit: {metrics.get('min_configured_temp')}°C")
mcol5.metric("ML Anomaly Rate", f"{metrics.get('anomaly_frequency_pct', 0.0)}%", f"{metrics.get('anomaly_count', 0)} observations")

sensor_breakdown = risk_data.get("sensor_breakdown", {})
if len(sensor_breakdown) > 1:
    st.markdown("#### 🏢 Individual Sensor Risk Breakdown")
    s_rows = []
    for s_id, s_info in sensor_breakdown.items():
        m = s_info.get("metrics", {})
        s_rows.append({
            "Sensor ID": s_id,
            "Risk Category": s_info.get("risk_category", "LOW"),
            "Risk Score": f"{s_info.get('risk_score', 0)}/100",
            "Peak Temp (°C)": m.get("max_recorded_temp", "N/A"),
            "Lowest Temp (°C)": m.get("min_recorded_temp", "N/A"),
            "Excursion Time (min)": m.get("excursion_duration_minutes", 0.0),
            "Excursion Episodes": m.get("num_excursion_episodes", 0),
            "Trend (°C/hr)": m.get("trend_deg_per_hour", 0.0)
        })
    st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)

st.markdown("---")

# ==========================================
# 2. REASONS & RECOMMENDATIONS
# ==========================================
rcol1, rcol2 = st.columns(2)

with rcol1:
    st.subheader("📋 Transparent Classification Factors")
    st.write("Why this shipment or cold-room received this risk classification:")
    for reason in reasons:
        st.markdown(f"- 📌 {reason}")

    # Point breakdown chart
    fig_pts = go.Figure(go.Bar(
        x=[breakdown.get("duration_points", 0), breakdown.get("severity_points", 0), breakdown.get("ml_anomaly_points", 0), breakdown.get("trend_points", 0)],
        y=["Duration Risk", "Peak Severity", "ML Anomalies", "Thermal Trend"],
        orientation="h",
        marker=dict(color=["#38BDF8", "#F59E0B", "#A855F7", "#10B981"])
    ))
    fig_pts.update_layout(xaxis_title="Points Added (Max 100)", yaxis_title="Factor")
    fig_pts = apply_plotly_theme(fig_pts, "Risk Factor Attribution")
    st.plotly_chart(fig_pts, use_container_width=True)

with rcol2:
    st.subheader("⚡ Recommended Operator Actions")
    st.write("Immediate mitigation steps based on observed risk triggers:")
    for i, rec in enumerate(recommendations, 1):
        st.markdown(
            f"""
            <div style="background-color: #12253B; border-left: 4px solid #38BDF8; padding: 12px 16px; margin-bottom: 10px; border-radius: 4px;">
                <span style="font-weight: 700; color: #F8FAFC;">Action {i}: </span>
                <span style="color: #E2E8F0;">{rec}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

# ==========================================
# 3. STRUCTURED OPERATIONAL ALERTS FEED
# ==========================================
st.markdown("---")
st.subheader("🚨 Generated Operational Alerts Feed")
st.caption("Triggered upon thermal exceedance, rapid temperature shifts, consecutive anomaly streaks, and ML breaches.")

alerts_df = generate_cold_chain_alerts(
    df,
    min_temp=st.session_state.min_temp,
    max_temp=st.session_state.max_temp
)

if alerts_df.empty:
    st.success("✅ Zero active alert triggers. All sensors conform to standard cold-chain parameters.")
else:
    # Filter by severity
    sev_filter = st.multiselect(
        "Filter Alerts by Severity",
        options=["CRITICAL", "WARNING", "INFO"],
        default=["CRITICAL", "WARNING", "INFO"]
    )
    filtered_alerts = alerts_df[alerts_df["severity"].isin(sev_filter)]

    st.dataframe(
        filtered_alerts[["timestamp", "sensor_id", "severity", "alert_type", "temperature", "anomaly_score", "reason", "recommended_action"]],
        use_container_width=True,
        hide_index=True
    )
