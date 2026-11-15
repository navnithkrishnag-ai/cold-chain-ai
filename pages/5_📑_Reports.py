"""
ColdGuard AI - PAGE 6: REPORTS & DATA EXPORTS
Enables downloading processed CSV files, anomaly logs, audit summary reports,
and reviewing sensor-wise operational performance.
"""

import streamlit as st
import pandas as pd
import config
from src.ui_components import inject_custom_css, initialize_session_state, render_sidebar, apply_plotly_theme
from src.report_generator import generate_sensor_summary, generate_audit_report_text
from src.risk_assessment import assess_cold_chain_risk
import plotly.express as px

st.set_page_config(
    page_title=f"{config.APP_NAME} - Reports & Exports",
    page_icon="📑",
    layout="wide"
)

inject_custom_css()
initialize_session_state()
render_sidebar()

st.title("📑 Cold-Chain Reports & Export Center")
st.markdown(
    "Export processed datasets with engineered time-series features, download anomaly audit logs, "
    "and inspect sensor-wise operational performance summaries."
)

df = st.session_state.df

if df.empty:
    st.warning("⚠️ No dataset loaded. Please go to **Data Upload** to load a dataset.")
    st.stop()

risk_eval = assess_cold_chain_risk(
    df,
    min_temp=st.session_state.min_temp,
    max_temp=st.session_state.max_temp,
    max_allowed_excursion_minutes=st.session_state.max_excursion_min
)
anomaly_summary = st.session_state.get("anomaly_summary", {})

# ==========================================
# 1. SENSOR-WISE ANALYSIS
# ==========================================
st.subheader("📊 Sensor-Wise Operational Analysis")
sensor_summary_df = generate_sensor_summary(df)

if not sensor_summary_df.empty:
    st.dataframe(sensor_summary_df, use_container_width=True, hide_index=True)

    # Visual chart
    c1, c2 = st.columns(2)
    with c1:
        fig_temp_bar = px.bar(
            sensor_summary_df,
            x="sensor_id",
            y="Temp Mean (°C)",
            error_y="Temp Std (°C)",
            color="Temp Mean (°C)",
            color_continuous_scale="Viridis",
            labels={"sensor_id": "Sensor", "Temp Mean (°C)": "Mean Temperature (°C)"}
        )
        fig_temp_bar = apply_plotly_theme(fig_temp_bar, "Mean Temperature & Variation by Sensor")
        st.plotly_chart(fig_temp_bar, use_container_width=True)

    with c2:
        fig_anom_bar = px.bar(
            sensor_summary_df,
            x="sensor_id",
            y="Anomalies",
            color="Anomaly Rate (%)",
            color_continuous_scale="Reds",
            labels={"sensor_id": "Sensor", "Anomalies": "Anomaly Count"}
        )
        fig_anom_bar = apply_plotly_theme(fig_anom_bar, "Detected Anomaly Count by Sensor")
        st.plotly_chart(fig_anom_bar, use_container_width=True)

st.markdown("---")

# ==========================================
# 2. DOWNLOADABLE EXPORT ARTIFACTS
# ==========================================
st.subheader("📥 Downloadable Reports & Datasets")

col1, col2, col3 = st.columns(3)

# 1. Processed CSV
with col1:
    st.markdown(
        """
        <div style="background-color: #12253B; border: 1px solid #2A4E77; border-radius: 8px; padding: 18px; height: 190px;">
            <h4 style="color: #38BDF8; margin-top: 0;">📊 Processed Time-Series CSV</h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Cleaned dataset containing rolling temperature averages, standard deviations, differences, and anomaly scores.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    processed_csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Processed CSV",
        data=processed_csv,
        file_name="coldguard_processed_data.csv",
        mime="text/csv",
        use_container_width=True
    )

# 2. Anomaly Report CSV
with col2:
    st.markdown(
        """
        <div style="background-color: #12253B; border: 1px solid #2A4E77; border-radius: 8px; padding: 18px; height: 190px;">
            <h4 style="color: #EF4444; margin-top: 0;">🚨 Anomaly Audit Log CSV</h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Filtered subset containing only anomalous readings, severity classifications, and plain-English reasons.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    anom_only = df[df.get("is_anomaly", pd.Series([False] * len(df)))].copy()
    anom_csv = anom_only.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=f"Download Anomaly Log ({len(anom_only)} rows)",
        data=anom_csv,
        file_name="coldguard_anomaly_report.csv",
        mime="text/csv",
        use_container_width=True
    )

# 3. Summary Audit Report
with col3:
    st.markdown(
        """
        <div style="background-color: #12253B; border: 1px solid #2A4E77; border-radius: 8px; padding: 18px; height: 190px;">
            <h4 style="color: #10B981; margin-top: 0;">📑 Formal Audit Report (.txt)</h4>
            <p style="font-size: 0.85rem; color: #94A3B8;">
                Comprehensive compliance summary with risk index, classification factors, and recommended operator actions.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    audit_text = generate_audit_report_text(risk_eval, anomaly_summary, df)
    st.download_button(
        label="Download Summary Audit Report",
        data=audit_text.encode("utf-8"),
        file_name="coldguard_audit_summary.txt",
        mime="text/plain",
        use_container_width=True
    )

# Preview of Summary Audit Report
with st.expander("👁️ Preview Formal Summary Audit Report"):
    st.code(audit_text, language="text")
