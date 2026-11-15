"""
ColdGuard AI - PAGE 2: DATA UPLOAD & INGESTION
Supports Zenodo 15130001 files, custom CSV uploads, delimiter detection,
data validation, missing value reports, and pre-bundled sample datasets.
"""

import streamlit as st
import pandas as pd
import config
from src.ui_components import inject_custom_css, initialize_session_state, render_sidebar
from src.data_loader import load_cold_storage_csv
from src.preprocessing import validate_and_clean_data
from src.feature_engineering import compute_cold_chain_features
from src.anomaly_detection import LocalAnomalyDetector

st.set_page_config(
    page_title=f"{config.APP_NAME} - Data Ingestion",
    page_icon="📁",
    layout="wide"
)

inject_custom_css()
initialize_session_state()
render_sidebar()

st.title("📁 Cold-Chain Data Upload & Diagnostics")
st.markdown(
    "Upload raw or preprocessed cold-storage CSV logs. The pipeline automatically inspects delimiters, "
    "identifies sensor IDs, cleans repeated headers, parses timestamps, and validates physical plausibility."
)

tab1, tab2 = st.tabs(["📤 Upload Custom CSV", "🧪 Load Demonstration Datasets"])

with tab1:
    uploaded_file = st.file_uploader(
        "Choose a cold-storage CSV file",
        type=["csv", "txt"],
        help="Upload semicolon-separated (Zenodo DHT22 format) or standard comma-separated time series."
    )

    if uploaded_file is not None:
        try:
            with st.spinner("Analyzing and parsing uploaded file..."):
                raw_df, meta = load_cold_storage_csv(
                    uploaded_file,
                    filename=uploaded_file.name
                )
            
            st.success(f"Successfully parsed **{uploaded_file.name}** using delimiter: `{meta['detected_delimiter']}`")

            # Ingestion Metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Raw Rows", f"{meta['rows_raw']:,}")
            mcol2.metric("Duplicate Headers Removed", meta["duplicate_headers_removed"])
            mcol3.metric("Sensors Identified", ", ".join(meta["sensors_found"]) or "SENSOR01")
            mcol4.metric("Cleaned Rows", f"{meta['rows_cleaned']:,}")

            # Cleaning & Range Validation
            clean_df, val_report = validate_and_clean_data(raw_df)

            col_val1, col_val2 = st.columns(2)
            with col_val1:
                st.subheader("📋 Ingestion Diagnostics")
                st.write({
                    "Detected Delimiter": meta["detected_delimiter"],
                    "Sensor ID(s)": val_report["sensors"],
                    "Missing Timestamps Dropped": val_report["missing_timestamps"],
                    "Missing Temperatures Handled": val_report["missing_temperature"],
                    "Out-of-Range Temperature Errors": val_report["out_of_range_temp_count"],
                    "Out-of-Range Humidity Errors": val_report["out_of_range_hum_count"]
                })

            with col_val2:
                st.subheader("🔍 Column Identification")
                col_types = pd.DataFrame({
                    "Column Name": clean_df.columns,
                    "Data Type": [str(t) for t in clean_df.dtypes],
                    "Non-Null Count": [clean_df[c].count() for c in clean_df.columns]
                })
                st.dataframe(col_types, use_container_width=True, hide_index=True)

            st.subheader("📄 Dataset Preview")
            st.dataframe(clean_df.head(15), use_container_width=True)

            if st.button("🚀 Process & Analyze This Dataset", type="primary"):
                with st.spinner("Computing backward-looking rolling features and anomaly scores..."):
                    feat_df = compute_cold_chain_features(clean_df)
                    detector = LocalAnomalyDetector(n_estimators=100)
                    res_df, summary = detector.fit_predict(
                        feat_df,
                        sigma_multiplier=st.session_state.sigma_threshold
                    )
                    st.session_state.df = res_df
                    st.session_state.anomaly_summary = summary
                    st.session_state.dataset_source_name = f"Uploaded: {uploaded_file.name}"

                st.success("✅ Features engineered and anomaly detection executed! Navigate to Temperature Monitoring or Anomaly Detection.")
                st.rerun()

        except Exception as e:
            st.error(f"Error parsing file: {str(e)}")

with tab2:
    st.markdown("### Pre-bundled Datasets for Immediate Demonstration")
    st.write(
        "You can explore the system right away using either the authentic Zenodo research extract or the synthetic simulation."
    )

    scol1, scol2 = st.columns(2)
    with scol1:
        st.markdown(
            """
            <div style="background-color: #12253B; border: 1px solid #2A4E77; border-radius: 8px; padding: 18px;">
                <h4 style="color: #38BDF8; margin-top: 0;">🔬 Zenodo Research Extract</h4>
                <p style="font-size: 0.88rem; color: #94A3B8;">
                    <strong>Source:</strong> Zenodo Record 15130001<br>
                    <strong>Sensors:</strong> SENSOR01, SENSOR03, SENSOR07 (DHT22)<br>
                    <strong>Format:</strong> Semicolon-delimited, dd.mm.yyyy, European numbers<br>
                    <strong>Events:</strong> Genuine door openings and cooling recovery.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Load Zenodo Research Sample", use_container_width=True):
            sample_path = config.SAMPLE_DATA_DIR / "zenodo_sample.csv"
            raw_df, _ = load_cold_storage_csv(sample_path, filename="zenodo_sample.csv")
            clean_df, _ = validate_and_clean_data(raw_df)
            feat_df = compute_cold_chain_features(clean_df)
            detector = LocalAnomalyDetector(n_estimators=100)
            res_df, summary = detector.fit_predict(feat_df, sigma_multiplier=st.session_state.sigma_threshold)
            
            st.session_state.df = res_df
            st.session_state.anomaly_summary = summary
            st.session_state.dataset_source_name = "Zenodo 15130001 Cold Storage (Research Extract)"
            st.success("Loaded Zenodo research extract!")
            st.rerun()

    with scol2:
        st.markdown(
            """
            <div style="background-color: #12253B; border: 1px solid #2A4E77; border-radius: 8px; padding: 18px;">
                <h4 style="color: #F59E0B; margin-top: 0;">🧪 Simulated Demonstration Dataset</h4>
                <p style="font-size: 0.88rem; color: #94A3B8;">
                    <strong>Label:</strong> [SIMULATED] Clearly identified synthetic data<br>
                    <strong>Sensors:</strong> SENSOR_SIM_A, SENSOR_SIM_B, SENSOR_SIM_C<br>
                    <strong>Profiles:</strong> Normal cycling, slow refrigerant drift, sudden door spikes, and sub-zero overcooling.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        if st.button("Load [SIMULATED] Demonstration Dataset", use_container_width=True):
            sample_path = config.SAMPLE_DATA_DIR / "synthetic_cold_chain.csv"
            raw_df, _ = load_cold_storage_csv(sample_path, filename="synthetic_cold_chain.csv")
            clean_df, _ = validate_and_clean_data(raw_df)
            feat_df = compute_cold_chain_features(clean_df)
            detector = LocalAnomalyDetector(n_estimators=100)
            res_df, summary = detector.fit_predict(feat_df, sigma_multiplier=st.session_state.sigma_threshold)
            
            st.session_state.df = res_df
            st.session_state.anomaly_summary = summary
            st.session_state.dataset_source_name = "[SIMULATED] Synthetic Cold-Chain Demo"
            st.success("Loaded synthetic demonstration dataset!")
            st.rerun()
