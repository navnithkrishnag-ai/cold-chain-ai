"""
ColdGuard AI - PAGE 4: AI ANOMALY DETECTION
Displays Random Cut Forest / Local ML anomaly scores, normal vs anomalous observations,
score distribution, anomaly timeline, configurable thresholding, and plain-English explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import config
from src.ui_components import inject_custom_css, initialize_session_state, render_sidebar, apply_plotly_theme
from src.anomaly_detection import LocalAnomalyDetector
from src.aws_s3 import S3Manager
from src.sagemaker_service import SageMakerRCFService

st.set_page_config(
    page_title=f"{config.APP_NAME} - AI Anomaly Detection",
    page_icon="🤖",
    layout="wide"
)

inject_custom_css()
initialize_session_state()
render_sidebar()

st.title("🤖 AI Anomaly Detection Engine")
st.markdown(
    "Unsupervised pattern discovery using tree-based anomaly detection. "
    "High anomaly scores isolate multi-dimensional environmental disruptions without relying on static thresholds."
)

df = st.session_state.df

if df.empty:
    st.warning("⚠️ No dataset loaded. Please go to **Data Upload** to load a dataset.")
    st.stop()

# Execution Mode Banner
current_mode = st.session_state.execution_mode
model_source_str = df.get("model_source", pd.Series(["LOCAL"])).iloc[0] if "model_source" in df else "LOCAL"

if current_mode == "LOCAL":
    st.info(
        f"🖥️ **Active Execution Mode: LOCAL DEMO (CPU)** | Model: {model_source_str}\n\n"
        "Calculates multi-dimensional anomaly scores using scikit-learn tree depth and statistical deviation."
    )
else:
    st.warning(
        f"☁️ **Active Execution Mode: AWS CLOUD** | Algorithm: Amazon SageMaker AI Random Cut Forest (RCF)\n\n"
        f"Target Region: `{config.AWS_REGION}` | S3 Bucket: `{config.S3_BUCKET or 'Not set'}`"
    )

# Execution Controls Card
with st.expander("⚙️ Anomaly Model Execution & AWS Cloud Actions", expanded=(current_mode == "AWS")):
    m_col1, m_col2 = st.columns([6, 6])
    
    with m_col1:
        st.subheader("🖥️ Run Local Anomaly Detection")
        st.write("Retrains the local model on the current dataset with the active sensitivity slider.")
        if st.button("Run / Recompute Local Anomaly Model", type="primary" if current_mode == "LOCAL" else "secondary"):
            with st.spinner("Fitting local anomaly model..."):
                detector = LocalAnomalyDetector(n_estimators=120)
                res_df, summary = detector.fit_predict(df, sigma_multiplier=st.session_state.sigma_threshold)
                st.session_state.df = res_df
                st.session_state.anomaly_summary = summary
                st.success(f"Local model executed! Flagged {summary['anomalies_detected']} anomalies ({summary['anomaly_rate_percent']}%).")
                st.rerun()

    with m_col2:
        st.subheader("☁️ Run AWS SageMaker AI RCF")
        st.write("Uploads data to Amazon S3, trains SageMaker Random Cut Forest, and retrieves batch inference scores.")
        
        # Test AWS Prerequisites before launching
        s3_mgr = S3Manager()
        sm_svc = SageMakerRCFService()
        prereqs = sm_svc.validate_cloud_prerequisites(s3_manager=s3_mgr)

        if not prereqs["ready"]:
            st.error("❌ AWS Cloud prerequisites not met:")
            for issue in prereqs["issues"]:
                st.markdown(f"- {issue}")
            with st.expander("📖 Beginner AWS Setup Instructions"):
                for inst in prereqs["instructions"]:
                    st.markdown(f"1. {inst}")
        else:
            st.success("✅ AWS credentials, S3 bucket, and IAM permissions verified.")
            if st.button("Execute SageMaker RCF Cloud Workflow", type="primary"):
                with st.spinner("Submitting SageMaker Training & Batch Transform jobs (this may take 3-5 minutes)..."):
                    try:
                        res_df, summary = sm_svc.run_rcf_workflow(
                            df,
                            s3_mgr,
                            sigma_multiplier=st.session_state.sigma_threshold
                        )
                        st.session_state.df = res_df
                        st.session_state.anomaly_summary = summary
                        st.success(f"AWS SageMaker inference complete! {summary['anomalies_detected']} anomalies identified.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"SageMaker execution error: {str(e)}")

st.markdown("---")

# Verify anomaly columns exist
if "anomaly_score" not in df.columns:
    detector = LocalAnomalyDetector(n_estimators=100)
    df, summary = detector.fit_predict(df, sigma_multiplier=st.session_state.sigma_threshold)
    st.session_state.df = df
    st.session_state.anomaly_summary = summary

score_mean = float(df["anomaly_score"].mean())
score_std = float(df["anomaly_score"].std())
threshold = round(score_mean + (st.session_state.sigma_threshold * score_std), 2)
df["anomaly_threshold"] = threshold
df["is_anomaly"] = df["anomaly_score"] >= threshold
total_anomalies = int(df["is_anomaly"].sum())

# Metrics Summary
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Mean Anomaly Score", f"{score_mean:.2f}")
mcol2.metric("Score Std Dev (σ)", f"{score_std:.2f}")
mcol3.metric("Applied Threshold (μ + kσ)", f"{threshold:.2f}")
mcol4.metric("Flagged Anomalies", f"{total_anomalies} ({(total_anomalies/len(df)*100):.1f}%)")

# ==========================================
# 1. ANOMALY TIMELINE CHART
# ==========================================
st.subheader("⏱️ Anomaly Score Trajectory & Threshold Cutoff")

fig_timeline = go.Figure()

# Background threshold line
fig_timeline.add_hline(
    y=threshold,
    line_dash="dash",
    line_color="#EF4444",
    annotation_text=f"Anomaly Threshold ({threshold:.2f})",
    annotation_position="top right"
)

# Plot anomaly score line
fig_timeline.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["anomaly_score"],
    mode="lines",
    name="Anomaly Score",
    line=dict(color="#38BDF8", width=1.5),
    hovertemplate="<b>%{x}</b><br>Score: %{y:.2f}<extra></extra>"
))

# Highlight anomalous observations with red dots
anom_pts = df[df["is_anomaly"]]
if not anom_pts.empty:
    fig_timeline.add_trace(go.Scatter(
        x=anom_pts["timestamp"],
        y=anom_pts["anomaly_score"],
        mode="markers",
        name="Anomalous Observations",
        marker=dict(color="#EF4444", size=7, symbol="diamond"),
        hovertemplate="<b>ANOMALY DETECTED</b><br>%{x}<br>Score: %{y:.2f}<extra></extra>"
    ))

fig_timeline.update_layout(
    xaxis_title="Timestamp",
    yaxis_title="Anomaly Score (Higher = More Abnormal)",
    hovermode="x unified"
)
fig_timeline = apply_plotly_theme(fig_timeline, "Anomaly Score Timeline vs Dynamic Heuristic Threshold")
st.plotly_chart(fig_timeline, use_container_width=True)

# ==========================================
# 2. SCORE DISTRIBUTION & SCATTER PLOT
# ==========================================
dcol1, dcol2 = st.columns(2)

with dcol1:
    st.subheader("📊 Anomaly Score Distribution")
    fig_hist = px.histogram(
        df,
        x="anomaly_score",
        nbins=40,
        color="is_anomaly",
        color_discrete_map={False: "#1E3E62", True: "#EF4444"},
        labels={"anomaly_score": "Score", "is_anomaly": "Is Anomaly"}
    )
    fig_hist.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"Cutoff: {threshold:.1f}"
    )
    fig_hist = apply_plotly_theme(fig_hist, "Distribution of Anomaly Scores (Normal vs Outliers)")
    st.plotly_chart(fig_hist, use_container_width=True)

with dcol2:
    st.subheader("🌡️ vs 💧 Temperature vs Humidity Feature Space")
    fig_scatter = px.scatter(
        df,
        x="temperature",
        y="humidity",
        color="anomaly_score",
        color_continuous_scale="Turbo",
        hover_data=["sensor_id", "temperature_change", "temperature_deviation"],
        labels={"temperature": "Temperature (°C)", "humidity": "Humidity (%)", "anomaly_score": "Score"}
    )
    fig_scatter = apply_plotly_theme(fig_scatter, "Observations Colored by Anomaly Score")
    st.plotly_chart(fig_scatter, use_container_width=True)

# ==========================================
# 3. EXPLANATION OF DETECTED ANOMALIES TABLE
# ==========================================
st.subheader("🔍 Plain-English Explanation of Detected Anomalies")

if anom_pts.empty:
    st.success("No anomalies detected at the current threshold.")
else:
    display_cols = ["timestamp", "sensor_id", "temperature", "humidity", "temperature_change", "anomaly_score", "anomaly_severity", "anomaly_reason"]
    avail_cols = [c for c in display_cols if c in anom_pts.columns]
    
    st.dataframe(
        anom_pts[avail_cols].head(50),
        use_container_width=True,
        hide_index=True
    )
