"""
ColdGuard AI - UI Components & Styling Module
Provides consistent navy-blue styling, KPI metric cards, Plotly layout themes,
and session state helpers across all Streamlit pages.
"""

from typing import Dict, Any, Optional
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import config
from src.data_loader import load_cold_storage_csv
from src.preprocessing import validate_and_clean_data
from src.feature_engineering import compute_cold_chain_features
from src.anomaly_detection import LocalAnomalyDetector
from src.risk_assessment import assess_cold_chain_risk


def inject_custom_css():
    """Injects high-polish navy-blue styling with restrained accent warning colors."""
    st.markdown(
        """
        <style>
        /* Main container polish */
        .main {
            background-color: #0B192C;
            color: #E2E8F0;
        }
        
        /* Metric Cards */
        .metric-card {
            background: linear-gradient(135deg, #12253B 0%, #1E3E62 100%);
            border: 1px solid #2A4E77;
            border-radius: 10px;
            padding: 18px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            margin-bottom: 15px;
        }
        .metric-label {
            font-size: 0.82rem;
            color: #94A3B8;
            text-transform: uppercase;
            font-weight: 600;
            letter-spacing: 0.05em;
        }
        .metric-value {
            font-size: 1.75rem;
            font-weight: 700;
            color: #F8FAFC;
            margin-top: 4px;
        }
        .metric-sub {
            font-size: 0.78rem;
            color: #38BDF8;
            margin-top: 2px;
        }
        
        /* Badges */
        .badge-local {
            background-color: #1E3E62;
            color: #38BDF8;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid #38BDF8;
            display: inline-block;
        }
        .badge-aws {
            background-color: #7C2D12;
            color: #FDBA74;
            padding: 4px 10px;
            border-radius: 12px;
            font-size: 0.75rem;
            font-weight: 600;
            border: 1px solid #F97316;
            display: inline-block;
        }
        .badge-risk-low {
            background-color: #064E3B;
            color: #6EE7B7;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.9rem;
        }
        .badge-risk-moderate {
            background-color: #78350F;
            color: #FDE68A;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.9rem;
        }
        .badge-risk-high {
            background-color: #7C2D12;
            color: #FDBA74;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.9rem;
        }
        .badge-risk-critical {
            background-color: #7F1D1D;
            color: #FCA5A5;
            padding: 4px 12px;
            border-radius: 6px;
            font-weight: 700;
            font-size: 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def initialize_session_state():
    """Initializes global state and preloads sample dataset if none is loaded."""
    if "df" not in st.session_state or st.session_state.df is None:
        # Load bundled Zenodo sample dataset as instant preview
        sample_path = config.SAMPLE_DATA_DIR / "zenodo_sample.csv"
        if sample_path.exists():
            df_raw, _ = load_cold_storage_csv(sample_path, filename="zenodo_sample.csv")
            df_clean, _ = validate_and_clean_data(df_raw)
            df_feat = compute_cold_chain_features(df_clean)
            
            # Run initial local model to provide live anomaly scores
            detector = LocalAnomalyDetector(n_estimators=100)
            df_result, summary = detector.fit_predict(df_feat, sigma_multiplier=config.DEFAULT_ANOMALY_SIGMA)
            
            st.session_state.df = df_result
            st.session_state.anomaly_summary = summary
            st.session_state.dataset_source_name = "Zenodo 15130001 Cold Storage (Sample Extract)"
        else:
            st.session_state.df = pd.DataFrame()
            st.session_state.anomaly_summary = {}
            st.session_state.dataset_source_name = "No dataset loaded"

    if "execution_mode" not in st.session_state:
        st.session_state.execution_mode = "LOCAL"

    if "min_temp" not in st.session_state:
        st.session_state.min_temp = config.DEFAULT_MIN_TEMP

    if "max_temp" not in st.session_state:
        st.session_state.max_temp = config.DEFAULT_MAX_TEMP

    if "max_excursion_min" not in st.session_state:
        st.session_state.max_excursion_min = config.DEFAULT_MAX_EXCURSION_MINUTES

    if "sigma_threshold" not in st.session_state:
        st.session_state.sigma_threshold = config.DEFAULT_ANOMALY_SIGMA


def render_sidebar():
    """Renders persistent brand header and global parameter controls in sidebar."""
    with st.sidebar:
        st.markdown(f"## {config.APP_ICON} {config.APP_NAME}")
        st.caption(config.APP_TAGLINE)
        st.markdown("---")

        # Execution Mode Indicator & Toggle
        st.subheader("⚙️ System Engine")
        mode = st.radio(
            "Execution Mode",
            options=["LOCAL", "AWS"],
            index=0 if st.session_state.execution_mode == "LOCAL" else 1,
            help="LOCAL: Runs offline ML models on laptop CPU. AWS: Connects to Amazon S3 & Amazon SageMaker AI."
        )
        st.session_state.execution_mode = mode

        if mode == "LOCAL":
            st.markdown('<span class="badge-local">● LOCAL DEMO MODE (Offline CPU)</span>', unsafe_allow_html=True)
            st.caption("Using genuine local scikit-learn anomaly scoring.")
        else:
            st.markdown('<span class="badge-aws">▲ AWS CLOUD MODE (SageMaker/S3)</span>', unsafe_allow_html=True)
            st.caption(f"Region: {config.AWS_REGION} | Bucket: {config.S3_BUCKET or 'Unconfigured'}")

        st.markdown("---")
        st.subheader("❄️ Thermal Preset")
        st.caption("Configurable operating band (Illustrative Chilled Goods Preset: 2°C to 8°C)")

        col1, col2 = st.columns(2)
        with col1:
            st.session_state.min_temp = st.number_input("Min Temp (°C)", value=st.session_state.min_temp, step=0.5)
        with col2:
            st.session_state.max_temp = st.number_input("Max Temp (°C)", value=st.session_state.max_temp, step=0.5)

        st.session_state.max_excursion_min = st.number_input(
            "Max Excursion (min)",
            value=st.session_state.max_excursion_min,
            step=5.0,
            help="Allowable cumulative time outside safe thermal band before severe risk classification."
        )

        st.session_state.sigma_threshold = st.slider(
            "Anomaly Sensitivity (k-Sigma)",
            min_value=1.5,
            max_value=4.5,
            value=float(st.session_state.sigma_threshold),
            step=0.25,
            help="Heuristic threshold multiplier: Threshold = Mean + (k * Std). Lower = more sensitive."
        )

        st.markdown("---")
        st.caption(f"Active Dataset:\n**{st.session_state.get('dataset_source_name', 'None')}**")


def apply_plotly_theme(fig: go.Figure, title: str = "") -> go.Figure:
    """Applies clean dark-navy styling to Plotly charts."""
    fig.update_layout(
        title={
            "text": title,
            "y": 0.95,
            "x": 0.05,
            "xanchor": "left",
            "yanchor": "top",
            "font": {"size": 16, "color": "#F8FAFC"}
        },
        paper_bgcolor="#0B192C",
        plot_bgcolor="#12253B",
        font={"color": "#94A3B8", "family": "Inter, sans-serif"},
        xaxis={
            "gridcolor": "#1E3E62",
            "showgrid": True,
            "zerolinecolor": "#2A4E77"
        },
        yaxis={
            "gridcolor": "#1E3E62",
            "showgrid": True,
            "zerolinecolor": "#2A4E77"
        },
        legend={
            "bgcolor": "rgba(18, 37, 59, 0.8)",
            "bordercolor": "#2A4E77",
            "borderwidth": 1
        },
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
