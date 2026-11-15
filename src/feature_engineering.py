"""
ColdGuard AI - Feature Engineering & Synthetic Data Generator Module
Computes time-series rolling features without future leakage,
and generates realistic simulated cold-chain datasets.
"""

from typing import List, Optional
import numpy as np
import pandas as pd


def compute_cold_chain_features(
    df: pd.DataFrame,
    window_size: int = 12
) -> pd.DataFrame:
    """
    Computes time-series features strictly per sensor and without future data leakage.
    
    Features produced:
    - timestamp
    - sensor_id
    - temperature
    - humidity
    - temperature_change: difference from previous observation (T_t - T_{t-1})
    - rolling_temperature_average: backward-looking rolling mean (window_size points)
    - rolling_temperature_std: backward-looking rolling standard deviation
    - temperature_deviation: current temperature minus rolling average
    """
    if df.empty:
        return df

    # Ensure chronological order
    featured = df.sort_values(by=["sensor_id", "timestamp"]).copy()

    # Per-sensor calculations strictly backward-looking
    featured["temperature_change"] = featured.groupby("sensor_id")["temperature"].diff().fillna(0.0)
    featured["rolling_temperature_average"] = featured.groupby("sensor_id")["temperature"].transform(
        lambda s: s.rolling(window=window_size, min_periods=1, center=False).mean()
    )
    featured["rolling_temperature_std"] = featured.groupby("sensor_id")["temperature"].transform(
        lambda s: s.rolling(window=window_size, min_periods=1, center=False).std().fillna(0.0)
    )
    featured["temperature_deviation"] = featured["temperature"] - featured["rolling_temperature_average"]
    featured = featured.reset_index(drop=True)

    # Standardize column order
    expected_order = [
        "timestamp",
        "sensor_id",
        "temperature",
        "humidity",
        "temperature_change",
        "rolling_temperature_average",
        "rolling_temperature_std",
        "temperature_deviation"
    ]
    other_cols = [c for c in featured.columns if c not in expected_order]
    return featured[expected_order + other_cols]


def generate_synthetic_dataset(
    n_points_per_sensor: int = 720,  # ~1 hour at 5-sec intervals or 6 hours at 30-sec
    start_time: str = "2024-09-02 08:00:00",
    freq: str = "30s"
) -> pd.DataFrame:
    """
    Generates a realistic synthetic cold-chain dataset clearly labeled as [SIMULATED].
    Simulates:
    1. Normal compressor cycling between 3.5°C and 5.5°C
    2. Gradual temperature drift (slow refrigerant leakage/insulation degradation)
    3. Sudden temperature spike (loading bay door left open)
    4. Sudden temperature drop (faulty thermostat / freezing risk)
    5. Realistic humidity variations (drying during freezing, spiking on open doors)
    """
    np.random.seed(42)
    timestamps = pd.date_range(start=start_time, periods=n_points_per_sensor, freq=freq)
    
    records: List[dict] = []

    # Sensor A: Standard storage zone with normal compressor cycling and an abrupt door opening
    # Sensor B: Back wall zone with gradual insulation degradation (slow creep upwards)
    # Sensor C: Near evaporator with an over-cooling anomaly (freeze risk)
    sensors = [
        ("SENSOR_SIM_A", "Door Bay Zone"),
        ("SENSOR_SIM_B", "Back Wall Zone"),
        ("SENSOR_SIM_C", "Evaporator Unit")
    ]

    for sensor_id, zone_desc in sensors:
        # Base refrigeration cycle (sinusoidal compressor oscillation around 4.5°C)
        t = np.linspace(0, 12 * np.pi, n_points_per_sensor)
        base_temp = 4.5 + 1.2 * np.sin(t) + np.random.normal(0, 0.15, n_points_per_sensor)
        base_hum = 75.0 + 5.0 * np.cos(t) + np.random.normal(0, 0.5, n_points_per_sensor)

        if sensor_id == "SENSOR_SIM_A":
            # Sudden temperature spike (Door opening between point 300 and 360)
            door_open_mask = (np.arange(n_points_per_sensor) >= 300) & (np.arange(n_points_per_sensor) <= 360)
            spike_profile = np.sin(np.linspace(0, np.pi, int(door_open_mask.sum()))) * 9.5
            base_temp[door_open_mask] += spike_profile
            # Humidity spikes when outside warm moist air rushes in
            base_hum[door_open_mask] += (spike_profile * 1.8)

        elif sensor_id == "SENSOR_SIM_B":
            # Gradual temperature drift (creeps up from point 400 to end due to slow gas leak)
            drift_mask = np.arange(n_points_per_sensor) >= 380
            n_drift = int(drift_mask.sum())
            drift_curve = np.linspace(0, 5.8, n_drift)
            base_temp[drift_mask] += drift_curve
            base_hum[drift_mask] -= (drift_curve * 1.5)

        elif sensor_id == "SENSOR_SIM_C":
            # Over-cooling drop (thermostat stuck ON between point 200 and 260, drops to -2.5°C)
            freeze_mask = (np.arange(n_points_per_sensor) >= 200) & (np.arange(n_points_per_sensor) <= 260)
            freeze_profile = np.sin(np.linspace(0, np.pi, int(freeze_mask.sum()))) * 7.2
            base_temp[freeze_mask] -= freeze_profile

        # Clamp humidity to physical bounds
        base_hum = np.clip(base_hum, 20.0, 99.0)

        for i in range(n_points_per_sensor):
            records.append({
                "timestamp": timestamps[i],
                "sensor_id": sensor_id,
                "temperature": round(float(base_temp[i]), 2),
                "humidity": round(float(base_hum[i]), 1),
                "data_source": "[SIMULATED] Synthetic Cold-Chain Demo",
                "zone_description": zone_desc
            })

    sim_df = pd.DataFrame(records)
    return compute_cold_chain_features(sim_df)
