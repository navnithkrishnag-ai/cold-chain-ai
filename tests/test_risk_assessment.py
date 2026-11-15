"""
Unit tests for cold-chain risk assessment and alert generation.
"""

import pytest
import pandas as pd
from src.risk_assessment import assess_cold_chain_risk
from src.alert_system import generate_cold_chain_alerts


def test_assess_cold_chain_risk_nominal():
    # Constant 4.5°C within [2.0, 8.0]
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-09-02 08:00:00", periods=50, freq="30s"),
        "sensor_id": ["S1"] * 50,
        "temperature": [4.5] * 50,
        "humidity": [75.0] * 50,
        "is_anomaly": [False] * 50
    })

    risk = assess_cold_chain_risk(df, min_temp=2.0, max_temp=8.0, max_allowed_excursion_minutes=30.0)
    assert risk["risk_category"] == "LOW"
    assert risk["risk_score"] < 25
    assert risk["metrics"]["percent_out_of_range"] == 0.0


def test_assess_cold_chain_risk_severe():
    # Temperature spikes to 22°C (severe excursion)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-09-02 08:00:00", periods=100, freq="1min"),
        "sensor_id": ["S1"] * 100,
        "temperature": [22.0] * 100,
        "humidity": [90.0] * 100,
        "is_anomaly": [True] * 100
    })

    risk = assess_cold_chain_risk(df, min_temp=2.0, max_temp=8.0, max_allowed_excursion_minutes=15.0)
    assert risk["risk_category"] == "CRITICAL"
    assert risk["risk_score"] >= 75
    assert risk["metrics"]["percent_out_of_range"] == 100.0


def test_generate_cold_chain_alerts():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-09-02 08:00:00", periods=5, freq="30s"),
        "sensor_id": ["S1"] * 5,
        "temperature": [4.0, 5.0, 15.0, 16.0, 16.0],  # rapid jump at index 2, out of range 2-4
        "humidity": [70.0, 70.0, 85.0, 85.0, 85.0],
        "temperature_change": [0.0, 1.0, 10.0, 1.0, 0.0],
        "anomaly_score": [12.0, 14.0, 95.0, 98.0, 92.0],
        "is_anomaly": [False, False, True, True, True]
    })

    alerts = generate_cold_chain_alerts(df, min_temp=2.0, max_temp=8.0, rapid_change_threshold=2.0)
    assert not alerts.empty
    alert_types = alerts["alert_type"].tolist()
    assert "High Temperature Excursion" in alert_types
    assert "Rapid Temperature Shift" in alert_types
