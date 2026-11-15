"""
Unit tests for data loading, delimiter detection, and preprocessing.
"""

import io
import pytest
import pandas as pd
from src.data_loader import detect_delimiter, load_cold_storage_csv
from src.preprocessing import validate_and_clean_data


def test_detect_delimiter():
    semicolon_sample = "Date;Time;Temperature;Humidity\n02.09.2024;09:26:30;5.3;82.1"
    comma_sample = "timestamp,sensor_id,temperature,humidity\n2024-09-02 09:26:30,S1,5.3,82.1"
    
    assert detect_delimiter(semicolon_sample) == ";"
    assert detect_delimiter(comma_sample) == ","


def test_load_zenodo_format_semicolon():
    raw_data = (
        "Date;Time;Temperature;Humidity;Sensor\n"
        "02.09.2024;09:26:30;5,3;82,1;SENSOR01\n"
        "02.09.2024;09:27:00;5,4;82,0;SENSOR01\n"
        "Date;Time;Temperature;Humidity;Sensor\n"  # Repeated header simulation
        "02.09.2024;09:27:30;5,5;81,9;SENSOR01\n"
    )
    buffer = io.StringIO(raw_data)
    df, meta = load_cold_storage_csv(buffer, filename="SENSOR01.csv")

    assert meta["detected_delimiter"] == ";"
    assert meta["duplicate_headers_removed"] == 1
    assert len(df) == 3
    assert df["sensor_id"].iloc[0] == "SENSOR01"
    assert df["temperature"].iloc[0] == pytest.approx(5.3)
    assert df["humidity"].iloc[0] == pytest.approx(82.1)
    assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])


def test_validate_and_clean_data():
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(["2024-09-02 09:00:00", "2024-09-02 09:00:30", "2024-09-02 09:01:00"]),
        "sensor_id": ["SENSOR01", "SENSOR01", "SENSOR01"],
        "temperature": [5.2, 999.0, 5.4],  # 999.0 is hardware error
        "humidity": [80.0, 81.0, 80.5]
    })

    cleaned, report = validate_and_clean_data(df, interpolate_missing=True)
    assert report["out_of_range_temp_count"] == 1
    # 999.0 should be interpolated between 5.2 and 5.4 -> 5.3
    assert len(cleaned) == 3
    assert cleaned["temperature"].iloc[1] == pytest.approx(5.3, abs=0.01)
