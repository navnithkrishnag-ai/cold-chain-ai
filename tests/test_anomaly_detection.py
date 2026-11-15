"""
Unit tests for local ML anomaly detection and heuristic thresholding.
"""

import pytest
import pandas as pd
from src.feature_engineering import generate_synthetic_dataset
from src.anomaly_detection import LocalAnomalyDetector


def test_local_anomaly_detector_training_and_scoring():
    # Use synthetic dataset with known injected spike
    df_sim = generate_synthetic_dataset(n_points_per_sensor=200)

    detector = LocalAnomalyDetector(n_estimators=100, random_state=42)
    result_df, summary = detector.fit_predict(df_sim, sigma_multiplier=3.0)

    assert "anomaly_score" in result_df.columns
    assert "anomaly_threshold" in result_df.columns
    assert "is_anomaly" in result_df.columns
    assert "anomaly_severity" in result_df.columns

    # Verify scores are positive floats
    assert (result_df["anomaly_score"] >= 0).all()
    assert summary["anomalies_detected"] > 0
    assert summary["score_mean"] > 0

    # Ensure anomalous rows receive explanation
    anomalies = result_df[result_df["is_anomaly"]]
    assert not anomalies.empty
    assert anomalies["anomaly_reason"].iloc[0] != "Normal environmental pattern"
