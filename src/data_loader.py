"""
ColdGuard AI - Data Loader Module
Handles loading and format detection for cold-storage environmental datasets,
including the Zenodo 15130001 research dataset, custom CSVs, and simulated data.
"""

import io
import re
from pathlib import Path
from typing import Tuple, Union, Optional
import pandas as pd


def detect_delimiter(sample_text: str) -> str:
    """
    Detect whether the file uses semicolon or comma as delimiter.
    Defaults to ';' if semicolon count exceeds comma count, else ','.
    """
    first_lines = sample_text.strip().splitlines()[:5]
    if not first_lines:
        return ","
    semicolons = sum(line.count(";") for line in first_lines)
    commas = sum(line.count(",") for line in first_lines)
    return ";" if semicolons >= commas else ","


def parse_datetime_column(df: pd.DataFrame) -> pd.Series:
    """
    Converts Date/Time or Timestamp columns to pandas datetime series.
    Supports:
    1. Separate 'Date' (dd.mm.yyyy or yyyy-mm-dd) and 'Time' (HH:MM:SS) columns (Zenodo format)
    2. Single 'timestamp', 'date_time', 'datetime' column
    """
    col_map = {col.lower().strip(): col for col in df.columns}

    # Case 1: Separate Date and Time columns
    date_col = next((col_map[k] for k in ["date", "datum"] if k in col_map), None)
    time_col = next((col_map[k] for k in ["time", "zeit"] if k in col_map), None)

    if date_col and time_col:
        combined_str = df[date_col].astype(str).str.strip() + " " + df[time_col].astype(str).str.strip()
        # Try European dd.mm.yyyy first (standard in Zenodo dataset)
        try:
            return pd.to_datetime(combined_str, format="%d.%m.%Y %H:%M:%S", errors="coerce")
        except Exception:
            return pd.to_datetime(combined_str, errors="coerce")

    # Case 2: Single timestamp column
    ts_col = next((col_map[k] for k in ["timestamp", "datetime", "date_time", "time", "date"] if k in col_map), None)
    if ts_col:
        try:
            # Check if format has dots (dd.mm.yyyy)
            sample_val = str(df[ts_col].dropna().iloc[0]) if not df[ts_col].dropna().empty else ""
            if "." in sample_val and len(sample_val.split(".")[0]) <= 2:
                return pd.to_datetime(df[ts_col], format="%d.%m.%Y %H:%M:%S", errors="coerce")
            return pd.to_datetime(df[ts_col], errors="coerce")
        except Exception:
            return pd.to_datetime(df[ts_col], errors="coerce")

    # Fallback: create index-based datetime starting from current time
    return pd.date_range(start="2024-09-02 09:00:00", periods=len(df), freq="5s")


def load_cold_storage_csv(
    source: Union[str, Path, io.BytesIO, io.StringIO],
    filename: Optional[str] = None,
    default_sensor_id: str = "SENSOR01"
) -> Tuple[pd.DataFrame, dict]:
    """
    Loads raw or preprocessed cold storage CSV data.
    
    Handles:
    - Semicolon or comma delimiters
    - Repeated header rows from logger restarts
    - Date and time formatting
    - Missing sensor IDs (inferred from filename or default)
    - Returns standardized DataFrame and metadata dictionary
    """
    metadata = {
        "source_name": filename or "Uploaded CSV",
        "detected_delimiter": ",",
        "rows_raw": 0,
        "rows_cleaned": 0,
        "duplicate_headers_removed": 0,
        "sensor_id": default_sensor_id,
        "sensors_found": []
    }

    # Extract sensor ID from filename if pattern matches (e.g. SENSOR01, SENSOR_01)
    if filename:
        sensor_match = re.search(r"(SENSOR\d{1,2}|SENSOR_\w+)", filename, re.IGNORECASE)
        if sensor_match:
            default_sensor_id = sensor_match.group(1).upper()
            metadata["sensor_id"] = default_sensor_id

    # Read content into text buffer to detect delimiter and filter repeated headers
    if isinstance(source, (str, Path)):
        with open(source, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
    elif isinstance(source, io.BytesIO):
        raw_text = source.getvalue().decode("utf-8", errors="replace")
    elif isinstance(source, io.StringIO):
        raw_text = source.getvalue()
    else:
        # Fallback for file-like objects
        content = source.read()
        if isinstance(content, bytes):
            raw_text = content.decode("utf-8", errors="replace")
        else:
            raw_text = str(content)

    delimiter = detect_delimiter(raw_text)
    metadata["detected_delimiter"] = delimiter

    # Read with pandas
    df_raw = pd.read_csv(io.StringIO(raw_text), sep=delimiter, dtype=str)
    metadata["rows_raw"] = len(df_raw)

    if df_raw.empty:
        return pd.DataFrame(), metadata

    # Find the header column names
    col_map_lower = {col.lower().strip(): col for col in df_raw.columns}

    # Filter out repeated header rows (e.g. where row value equals column name)
    primary_col = df_raw.columns[0]
    is_repeated_header = df_raw[primary_col].astype(str).str.strip().str.lower() == primary_col.strip().lower()
    metadata["duplicate_headers_removed"] = int(is_repeated_header.sum())
    df_cleaned = df_raw[~is_repeated_header].copy()

    # Identify temperature and humidity columns
    temp_col = next((col_map_lower[k] for k in ["temperature", "temp", "t", "celsius"] if k in col_map_lower), None)
    hum_col = next((col_map_lower[k] for k in ["humidity", "hum", "rh", "relative_humidity"] if k in col_map_lower), None)
    sensor_col = next((col_map_lower[k] for k in ["sensor", "sensor_id", "sensorid", "device"] if k in col_map_lower), None)

    # Standardize Sensor ID
    if sensor_col and sensor_col in df_cleaned.columns:
        df_cleaned["sensor_id"] = df_cleaned[sensor_col].astype(str).str.strip()
    else:
        df_cleaned["sensor_id"] = default_sensor_id

    metadata["sensors_found"] = df_cleaned["sensor_id"].unique().tolist()

    # Standardize Timestamp
    df_cleaned["timestamp"] = parse_datetime_column(df_cleaned)

    # Parse numeric temperature and humidity (support European comma as decimal e.g. 5,4 -> 5.4)
    if temp_col and temp_col in df_cleaned.columns:
        temp_series = df_cleaned[temp_col].astype(str).str.replace(",", ".").str.strip()
        df_cleaned["temperature"] = pd.to_numeric(temp_series, errors="coerce")
    else:
        df_cleaned["temperature"] = float("nan")

    if hum_col and hum_col in df_cleaned.columns:
        hum_series = df_cleaned[hum_col].astype(str).str.replace(",", ".").str.strip()
        df_cleaned["humidity"] = pd.to_numeric(hum_series, errors="coerce")
    else:
        df_cleaned["humidity"] = float("nan")

    metadata["rows_cleaned"] = len(df_cleaned)
    return df_cleaned, metadata
