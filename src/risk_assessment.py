"""
ColdGuard AI - Cold-Chain Risk Assessment Engine
Rule-based evaluation of thermal integrity, excursion episodes, and machine learning anomalies.

Calculates metrics independently per sensor:
- Sorts each sensor's observations chronologically.
- Computes temperature trends within each sensor.
- Detects excursion episodes within each sensor.
- Estimates excursion duration using actual timestamps and sampling intervals.
- Provides a detailed sensor-wise risk breakdown.
- Determines overall cold-chain risk using the highest individual sensor risk.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


RISK_RANKS = {
    "LOW": 1,
    "MODERATE": 2,
    "HIGH": 3,
    "CRITICAL": 4
}


def _assess_single_sensor_risk(
    sensor_df: pd.DataFrame,
    sensor_id: str,
    min_temp: float = 2.0,
    max_temp: float = 8.0,
    max_allowed_excursion_minutes: float = 30.0
) -> Dict[str, Any]:
    """
    Evaluates cold-chain risk independently for a single sensor.
    Assumes sensor_df is already sorted chronologically.
    """
    n_points = len(sensor_df)
    if n_points == 0:
        return {
            "sensor_id": sensor_id,
            "risk_category": "LOW",
            "risk_score": 0,
            "reasons": ["No records for this sensor."],
            "point_breakdown": {"duration_points": 0, "severity_points": 0, "ml_anomaly_points": 0, "trend_points": 0},
            "metrics": {}
        }

    temp = sensor_df["temperature"].values

    # Excursion boolean masks
    high_excursion = temp > max_temp
    low_excursion = temp < min_temp
    any_excursion = high_excursion | low_excursion

    max_recorded = float(np.max(temp))
    min_recorded = float(np.min(temp))
    mean_temp = float(np.mean(temp))
    out_of_range_count = int(any_excursion.sum())
    percent_out_of_range = round((out_of_range_count / max(n_points, 1)) * 100.0, 2)

    # Calculate sampling intervals & excursion duration from actual timestamps
    has_datetime = (
        "timestamp" in sensor_df.columns
        and pd.api.types.is_datetime64_any_dtype(sensor_df["timestamp"])
    )

    if has_datetime and n_points >= 2:
        time_deltas_sec = sensor_df["timestamp"].diff().dt.total_seconds().abs()
        median_step_sec = float(time_deltas_sec.dropna().median())
        if np.isnan(median_step_sec) or median_step_sec <= 0:
            median_step_sec = 30.0
        
        # Fill first reading's delta with median step
        step_durations_sec = time_deltas_sec.fillna(median_step_sec)
        # Sum actual elapsed seconds where temperature was out of range
        excursion_seconds = float(step_durations_sec[any_excursion].sum())
        excursion_duration_minutes = round(excursion_seconds / 60.0, 1)
        total_duration_minutes = round(float((sensor_df["timestamp"].iloc[-1] - sensor_df["timestamp"].iloc[0]).total_seconds()) / 60.0, 1)
    else:
        median_step_sec = 30.0
        total_duration_minutes = round((n_points * 30.0) / 60.0, 1)
        excursion_duration_minutes = round((out_of_range_count * 30.0) / 60.0, 1)

    # Excursion Episodes Count: continuous blocks within this sensor
    transitions = np.diff(any_excursion.astype(int), prepend=0)
    num_excursion_episodes = int((transitions == 1).sum())

    # ML Anomaly frequency for this sensor
    has_ml_scores = "is_anomaly" in sensor_df.columns
    if has_ml_scores:
        anomaly_count = int(sensor_df["is_anomaly"].sum())
        anomaly_frequency_pct = round((anomaly_count / max(n_points, 1)) * 100.0, 2)
    else:
        anomaly_count = 0
        anomaly_frequency_pct = 0.0

    # Temperature Trend (°C / hour) within this sensor
    trend_deg_per_hour = 0.0
    if n_points >= 3:
        if has_datetime and total_duration_minutes > 0:
            time_hours = (sensor_df["timestamp"] - sensor_df["timestamp"].iloc[0]).dt.total_seconds() / 3600.0
            if time_hours.max() > 0:
                slope, _ = np.polyfit(time_hours.values, temp, 1)
                trend_deg_per_hour = round(float(slope), 2)
        else:
            step_hours = median_step_sec / 3600.0
            x_hours = np.arange(n_points) * step_hours
            if x_hours[-1] > 0:
                slope, _ = np.polyfit(x_hours, temp, 1)
                trend_deg_per_hour = round(float(slope), 2)

    # ==========================================
    # RULE-BASED SCORING FOR SENSOR
    # ==========================================
    reasons: List[str] = []
    point_breakdown: Dict[str, float] = {}

    # Component 1: Excursion Duration (up to 40 points)
    duration_ratio = excursion_duration_minutes / max(max_allowed_excursion_minutes, 1.0)
    if duration_ratio <= 0.0:
        duration_pts = 0.0
    elif duration_ratio <= 0.5:
        duration_pts = 10.0
        reasons.append(f"Brief excursion duration ({excursion_duration_minutes} min, within safe limit of {max_allowed_excursion_minutes} min).")
    elif duration_ratio <= 1.0:
        duration_pts = 25.0
        reasons.append(f"Excursion duration approaching maximum threshold ({excursion_duration_minutes} / {max_allowed_excursion_minutes} min).")
    elif duration_ratio <= 2.0:
        duration_pts = 35.0
        reasons.append(f"Excursion duration exceeded allowed limit ({excursion_duration_minutes} min > {max_allowed_excursion_minutes} min).")
    else:
        duration_pts = 40.0
        reasons.append(f"Severe extended excursion duration ({excursion_duration_minutes} min, > 2x limit).")
    point_breakdown["duration_points"] = duration_pts

    # Component 2: Temperature Severity Exceedance (up to 30 points)
    severity_pts = 0.0
    if max_recorded > (max_temp + 8.0):
        severity_pts += 25.0
        reasons.append(f"Extreme heat excursion: peak temp {max_recorded:.1f}°C (+{(max_recorded - max_temp):.1f}°C over max {max_temp}°C).")
    elif max_recorded > (max_temp + 3.0):
        severity_pts += 15.0
        reasons.append(f"Significant temperature elevation: peak {max_recorded:.1f}°C (exceeds {max_temp}°C).")
    elif max_recorded > max_temp:
        severity_pts += 8.0
        reasons.append(f"Minor high temperature excursion: peak {max_recorded:.1f}°C.")

    # Freezing risk
    if min_recorded < 0.0:
        severity_pts += 25.0
        reasons.append(f"Critical freezing hazard: sub-zero reading {min_recorded:.1f}°C detected.")
    elif min_recorded < min_temp:
        severity_pts += 10.0
        reasons.append(f"Under-temperature excursion: dropped to {min_recorded:.1f}°C (below min {min_temp}°C).")

    severity_pts = min(severity_pts, 30.0)
    point_breakdown["severity_points"] = severity_pts

    # Component 3: ML Anomaly Frequency (up to 20 points)
    ml_pts = 0.0
    if has_ml_scores:
        if anomaly_frequency_pct > 15.0:
            ml_pts = 20.0
            reasons.append(f"High frequency of ML-detected unusual patterns ({anomaly_frequency_pct}% anomalous).")
        elif anomaly_frequency_pct > 5.0:
            ml_pts = 12.0
            reasons.append(f"Moderate ML anomaly activity detected ({anomaly_frequency_pct}% anomalous).")
        elif anomaly_frequency_pct > 1.0:
            ml_pts = 5.0
            reasons.append(f"Occasional ML anomalies detected ({anomaly_frequency_pct}%).")
        else:
            reasons.append("ML anomaly patterns are within normal baseline range.")
    else:
        reasons.append("ML anomaly scores not yet calculated; risk relies solely on thermal rules.")
    point_breakdown["ml_anomaly_points"] = ml_pts

    # Component 4: Thermal Rate of Rise / Trend (up to 10 points)
    trend_pts = 0.0
    if trend_deg_per_hour >= 2.0:
        trend_pts = 10.0
        reasons.append(f"Rapid warming trend detected (+{trend_deg_per_hour}°C/hour).")
    elif trend_deg_per_hour >= 0.75:
        trend_pts = 5.0
        reasons.append(f"Gradual upward thermal drift detected (+{trend_deg_per_hour}°C/hour).")
    point_breakdown["trend_points"] = trend_pts

    total_score = min(int(round(duration_pts + severity_pts + ml_pts + trend_pts)), 100)

    if total_score >= 75:
        category = "CRITICAL"
    elif total_score >= 50:
        category = "HIGH"
    elif total_score >= 25:
        category = "MODERATE"
    else:
        category = "LOW"

    return {
        "sensor_id": sensor_id,
        "risk_category": category,
        "risk_score": total_score,
        "point_breakdown": point_breakdown,
        "reasons": reasons,
        "metrics": {
            "sensor_id": sensor_id,
            "observations": n_points,
            "min_configured_temp": min_temp,
            "max_configured_temp": max_temp,
            "max_allowed_excursion_minutes": max_allowed_excursion_minutes,
            "max_recorded_temp": round(max_recorded, 2),
            "min_recorded_temp": round(min_recorded, 2),
            "mean_recorded_temp": round(mean_temp, 2),
            "num_excursion_episodes": num_excursion_episodes,
            "excursion_duration_minutes": excursion_duration_minutes,
            "total_duration_minutes": total_duration_minutes,
            "percent_out_of_range": percent_out_of_range,
            "has_ml_scores": has_ml_scores,
            "anomaly_frequency_pct": anomaly_frequency_pct,
            "anomaly_count": anomaly_count,
            "trend_deg_per_hour": trend_deg_per_hour
        }
    }


def assess_cold_chain_risk(
    df: pd.DataFrame,
    min_temp: float = 2.0,
    max_temp: float = 8.0,
    max_allowed_excursion_minutes: float = 30.0
) -> Dict[str, Any]:
    """
    Evaluates cold-chain environmental data against product thermal specifications.
    
    Multi-sensor processing:
    1. Evaluates each sensor independently chronologically.
    2. Constructs a sensor-wise risk breakdown.
    3. Calculates overall facility risk using the highest individual sensor risk.
    """
    if df.empty or "temperature" not in df.columns:
        return {
            "risk_category": "UNKNOWN",
            "risk_score": 0,
            "reasons": ["No data provided for evaluation."],
            "recommendations": ["Upload or provide environmental dataset."],
            "metrics": {},
            "sensor_breakdown": {}
        }

    # Identify sensors
    if "sensor_id" in df.columns:
        sensor_ids = df["sensor_id"].unique().tolist()
    else:
        sensor_ids = ["SENSOR01"]
        df = df.copy()
        df["sensor_id"] = "SENSOR01"

    sensor_breakdown: Dict[str, Any] = {}
    highest_rank = 0
    highest_risk_score = 0
    driver_sensor = sensor_ids[0]

    for s_id in sensor_ids:
        # Filter and sort each sensor chronologically
        s_df = df[df["sensor_id"] == s_id].copy()
        if "timestamp" in s_df.columns:
            s_df = s_df.sort_values(by="timestamp").reset_index(drop=True)
        else:
            s_df = s_df.reset_index(drop=True)

        res = _assess_single_sensor_risk(
            sensor_df=s_df,
            sensor_id=str(s_id),
            min_temp=min_temp,
            max_temp=max_temp,
            max_allowed_excursion_minutes=max_allowed_excursion_minutes
        )
        sensor_breakdown[str(s_id)] = res

        s_rank = RISK_RANKS.get(res["risk_category"], 1)
        s_score = res["risk_score"]

        # Track the highest individual sensor risk
        if (s_rank > highest_rank) or (s_rank == highest_rank and s_score > highest_risk_score):
            highest_rank = s_rank
            highest_risk_score = s_score
            driver_sensor = str(s_id)

    driver_res = sensor_breakdown[driver_sensor]
    overall_category = driver_res["risk_category"]
    overall_score = driver_res["risk_score"]

    # Recommendations based on overall risk
    if overall_category == "CRITICAL":
        recommendations = [
            f"Immediate facility alarm: Sensor '{driver_sensor}' detected critical thermal excursion.",
            "Verify refrigeration compressor status and check for door seals or power disruption.",
            "Quarantine sensitive goods located in affected zones for quality assurance testing before release.",
            "Review thermal exposure log for regulatory compliance reporting."
        ]
    elif overall_category == "HIGH":
        recommendations = [
            f"High risk detected on sensor '{driver_sensor}'.",
            "Inspect cold storage unit and check physical door closure and loading activity.",
            "Verify thermostat settings and air circulation around sensor locations.",
            "Monitor product temperatures closely over the next operational shift."
        ]
    elif overall_category == "MODERATE":
        recommendations = [
            f"Moderate risk observed on sensor '{driver_sensor}'.",
            "Investigate brief excursion causes (e.g. routine door openings during restocking).",
            "Ensure refrigeration doors remain closed when not in active use.",
            "Observe subsequent readings to confirm return to standard operating range."
        ]
    else:
        recommendations = [
            "Maintain standard monitoring routine.",
            "All sensors remain well within acceptable operating band.",
            "No immediate operator action required."
        ]

    # Prefix driver reasons with sensor name if multi-sensor
    overall_reasons = []
    if len(sensor_ids) > 1:
        overall_reasons.append(f"Highest risk identified at sensor '{driver_sensor}' ({overall_category} RISK, Score: {overall_score}/100).")
    for r in driver_res["reasons"]:
        overall_reasons.append(f"[{driver_sensor}] {r}" if len(sensor_ids) > 1 else r)

    # Consolidated metrics
    all_metrics = [v["metrics"] for v in sensor_breakdown.values() if "metrics" in v and v["metrics"]]
    consolidated_metrics = {
        "min_configured_temp": min_temp,
        "max_configured_temp": max_temp,
        "max_allowed_excursion_minutes": max_allowed_excursion_minutes,
        "driver_sensor": driver_sensor,
        "total_sensors_evaluated": len(sensor_ids),
        "max_recorded_temp": max(m["max_recorded_temp"] for m in all_metrics) if all_metrics else 0.0,
        "min_recorded_temp": min(m["min_recorded_temp"] for m in all_metrics) if all_metrics else 0.0,
        "mean_recorded_temp": round(float(np.mean([m["mean_recorded_temp"] for m in all_metrics])), 2) if all_metrics else 0.0,
        "num_excursion_episodes": sum(m["num_excursion_episodes"] for m in all_metrics) if all_metrics else 0,
        "excursion_duration_minutes": driver_res["metrics"].get("excursion_duration_minutes", 0.0),
        "percent_out_of_range": driver_res["metrics"].get("percent_out_of_range", 0.0),
        "has_ml_scores": driver_res["metrics"].get("has_ml_scores", False),
        "anomaly_frequency_pct": driver_res["metrics"].get("anomaly_frequency_pct", 0.0),
        "anomaly_count": sum(m.get("anomaly_count", 0) for m in all_metrics) if all_metrics else 0,
        "trend_deg_per_hour": driver_res["metrics"].get("trend_deg_per_hour", 0.0)
    }

    return {
        "risk_category": overall_category,
        "risk_score": overall_score,
        "point_breakdown": driver_res["point_breakdown"],
        "reasons": overall_reasons,
        "recommendations": recommendations,
        "metrics": consolidated_metrics,
        "sensor_breakdown": sensor_breakdown
    }
