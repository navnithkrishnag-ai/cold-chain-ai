"""
Unit tests for time-series feature engineering and synthetic data generation.
"""

import pytest
import pandas as pd
from src.feature_engineering import compute_cold_chain_features, generate_synthetic_dataset


def test_compute_cold_chain_features_no_future_leak():
    # Construct sequence where sudden jump happens at index 3
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-09-02 09:00:00", periods=5, freq="30s"),
        "sensor_id": ["S1", "S1", "S1", "S1", "S1"],
        "temperature": [4.0, 4.0, 4.0, 10.0, 10.0],
        "humidity": [80.0, 80.0, 80.0, 80.0, 80.0]
    })

    feat = compute_cold_chain_features(df, window_size=3)

    # Required columns check
    expected_cols = [
        "timestamp", "sensor_id", "temperature", "humidity",
        "temperature_change", "rolling_temperature_average",
        "rolling_temperature_std", "temperature_deviation"
    ]
    for col in expected_cols:
        assert col in feat.columns

    # Verify backward-looking calculation at index 2 (before jump):
    # Rolling avg of [4.0, 4.0, 4.0] must be exactly 4.0
    assert feat["rolling_temperature_average"].iloc[2] == pytest.approx(4.0)
    assert feat["temperature_change"].iloc[2] == pytest.approx(0.0)

    # Index 3 jump to 10.0: change should be +6.0
    assert feat["temperature_change"].iloc[3] == pytest.approx(6.0)
    # Rolling average at index 3 over window=3 should be (4.0 + 4.0 + 10.0) / 3 = 6.0
    assert feat["rolling_temperature_average"].iloc[3] == pytest.approx(6.0)
    # Temperature deviation = 10.0 - 6.0 = 4.0
    assert feat["temperature_deviation"].iloc[3] == pytest.approx(4.0)


def test_generate_synthetic_dataset():
    df_sim = generate_synthetic_dataset(n_points_per_sensor=100)
    assert len(df_sim) == 300  # 3 sensors x 100 points
    assert "[SIMULATED]" in df_sim["data_source"].iloc[0]
    assert set(df_sim["sensor_id"].unique()) == {"SENSOR_SIM_A", "SENSOR_SIM_B", "SENSOR_SIM_C"}
