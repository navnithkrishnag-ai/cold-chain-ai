"""
ColdGuard AI - Preprocessing Module
Cleans environmental readings, validates sensor ranges, handles missing values,
and prepares datasets for time-series feature engineering and anomaly detection.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


# Physical plausibility boundaries for DHT22 cold storage sensors
MIN_VALID_TEMP = -40.0  # °C
MAX_VALID_TEMP = 70.0   # °C
MIN_VALID_HUM = 0.0     # %
MAX_VALID_HUM = 100.0   # %


def validate_and_clean_data(
    df: pd.DataFrame,
    interpolate_missing: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw cold-storage records:
    1. Validates required columns (timestamp, sensor_id, temperature, humidity)
    2. Drops rows with invalid or missing timestamps
    3. Detects and handles invalid sensor readings (out-of-range sensor errors)
    4. Sorts by sensor_id and timestamp
    5. Optionally interpolates short missing segments per sensor
    6. Returns cleaned DataFrame and comprehensive validation report
    """
    report: Dict[str, Any] = {
        "initial_rows": len(df),
        "missing_timestamps": 0,
        "missing_temperature": 0,
        "missing_humidity": 0,
        "out_of_range_temp_count": 0,
        "out_of_range_hum_count": 0,
        "final_rows": 0,
        "sensors": []
    }

    if df.empty:
        return df, report

    # Make working copy
    cleaned = df.copy()

    # Verify essential columns
    for col in ["timestamp", "sensor_id", "temperature", "humidity"]:
        if col not in cleaned.columns:
            raise ValueError(f"Required column '{col}' missing from input data.")

    # Drop NaT timestamps
    nat_mask = cleaned["timestamp"].isna()
    report["missing_timestamps"] = int(nat_mask.sum())
    cleaned = cleaned[~nat_mask].copy()

    # Count initial NaNs
    report["missing_temperature"] = int(cleaned["temperature"].isna().sum())
    report["missing_humidity"] = int(cleaned["humidity"].isna().sum())

    # Detect sensor hardware anomalies / out-of-range readings
    temp_invalid = (cleaned["temperature"] < MIN_VALID_TEMP) | (cleaned["temperature"] > MAX_VALID_TEMP)
    hum_invalid = (cleaned["humidity"] < MIN_VALID_HUM) | (cleaned["humidity"] > MAX_VALID_HUM)

    report["out_of_range_temp_count"] = int(temp_invalid.sum())
    report["out_of_range_hum_count"] = int(hum_invalid.sum())

    # Coerce out-of-range readings to NaN so they can be handled cleanly
    cleaned.loc[temp_invalid, "temperature"] = np.nan
    cleaned.loc[hum_invalid, "humidity"] = np.nan

    # Sort chronologically by sensor and timestamp
    cleaned = cleaned.sort_values(by=["sensor_id", "timestamp"]).reset_index(drop=True)

    # Remove duplicate timestamp rows per sensor
    cleaned = cleaned.drop_duplicates(subset=["sensor_id", "timestamp"]).reset_index(drop=True)

    # Handle missing values
    if interpolate_missing:
        # Interpolate linearly per sensor with max limit of 5 consecutive missing steps
        cleaned["temperature"] = cleaned.groupby("sensor_id")["temperature"].transform(
            lambda s: s.interpolate(method="linear", limit=5)
        )
        cleaned["humidity"] = cleaned.groupby("sensor_id")["humidity"].transform(
            lambda s: s.interpolate(method="linear", limit=5)
        )

    # Drop remaining rows where temperature or humidity are still NaN
    cleaned = cleaned.dropna(subset=["temperature", "humidity"]).reset_index(drop=True)

    report["final_rows"] = len(cleaned)
    report["sensors"] = cleaned["sensor_id"].unique().tolist()

    return cleaned, report
