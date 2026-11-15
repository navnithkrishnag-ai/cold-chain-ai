"""
Unit tests for cold-chain risk assessment and alert generation.
Includes multi-sensor independent calculations, per-sensor excursion episodes,
per-sensor temperature trends, and overall risk aggregation.
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


def test_multi_sensor_excursion_episodes():
    # Create interleaved records for S1 and S2
    # S1 has 1 continuous excursion episode: [normal, normal, EXCURSION, EXCURSION, normal] -> 1 episode
    # S2 has 2 separate excursion episodes: [EXCURSION, normal, EXCURSION, normal, normal] -> 2 episodes
    ts = pd.date_range("2024-09-02 08:00:00", periods=5, freq="1min")
    
    rows = []
    s1_temps = [4.0, 4.0, 12.0, 14.0, 4.0]
    s2_temps = [15.0, 4.0, 16.0, 4.0, 4.0]

    for i in range(5):
        rows.append({"timestamp": ts[i], "sensor_id": "S1", "temperature": s1_temps[i], "humidity": 75.0})
        rows.append({"timestamp": ts[i], "sensor_id": "S2", "temperature": s2_temps[i], "humidity": 75.0})

    df = pd.DataFrame(rows)
    # Shuffle or interleave rows
    risk = assess_cold_chain_risk(df, min_temp=2.0, max_temp=8.0)

    breakdown = risk["sensor_breakdown"]
    assert "S1" in breakdown
    assert "S2" in breakdown
    assert breakdown["S1"]["metrics"]["num_excursion_episodes"] == 1
    assert breakdown["S2"]["metrics"]["num_excursion_episodes"] == 2


def test_per_sensor_temperature_trends():
    # S1 warms up from 4°C to 10°C over 1 hour
    # S2 cools down from 8°C to 2°C over 1 hour
    ts = pd.date_range("2024-09-02 08:00:00", periods=7, freq="10min")
    
    rows = []
    s1_temps = [4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    s2_temps = [8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]

    for i in range(7):
        rows.append({"timestamp": ts[i], "sensor_id": "S1", "temperature": s1_temps[i], "humidity": 75.0})
        rows.append({"timestamp": ts[i], "sensor_id": "S2", "temperature": s2_temps[i], "humidity": 75.0})

    df = pd.DataFrame(rows)
    risk = assess_cold_chain_risk(df, min_temp=2.0, max_temp=8.0)

    s1_trend = risk["sensor_breakdown"]["S1"]["metrics"]["trend_deg_per_hour"]
    s2_trend = risk["sensor_breakdown"]["S2"]["metrics"]["trend_deg_per_hour"]

    assert s1_trend > 0, "S1 should have positive warming trend"
    assert s2_trend < 0, "S2 should have negative cooling trend"


def test_overall_risk_aggregates_highest_sensor():
    # S1 is completely nominal (LOW risk)
    # S2 experiences massive 25°C excursion (CRITICAL risk)
    ts = pd.date_range("2024-09-02 08:00:00", periods=30, freq="1min")
    rows = []
    for t in ts:
        rows.append({"timestamp": t, "sensor_id": "S1", "temperature": 4.5, "humidity": 75.0, "is_anomaly": False})
        rows.append({"timestamp": t, "sensor_id": "S2", "temperature": 25.0, "humidity": 75.0, "is_anomaly": True})

    df = pd.DataFrame(rows)
    risk = assess_cold_chain_risk(df, min_temp=2.0, max_temp=8.0, max_allowed_excursion_minutes=10.0)

    assert risk["sensor_breakdown"]["S1"]["risk_category"] == "LOW"
    assert risk["sensor_breakdown"]["S2"]["risk_category"] == "CRITICAL"
    # Overall risk must equal the highest sensor risk (CRITICAL)
    assert risk["risk_category"] == "CRITICAL"
    assert risk["risk_score"] == risk["sensor_breakdown"]["S2"]["risk_score"]


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
