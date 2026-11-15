"""
ColdGuard AI - Cold-Chain Risk Assessment Engine
Rule-based evaluation of thermal integrity, excursion episodes, and machine learning anomalies.
NOTE: This is an explicit rule-based risk evaluation system designed for early warning.
It is NOT a predictive ML model claiming certainty of spoilage or mechanical failure.
"""

from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def assess_cold_chain_risk(
    df: pd.DataFrame,
    min_temp: float = 2.0,
    max_temp: float = 8.0,
    max_allowed_excursion_minutes: float = 30.0
) -> Dict[str, Any]:
    """
    Evaluates cold-chain environmental data against product thermal specifications.
    
    Parameters:
    - min_temp: Lower thermal threshold (e.g. 2.0°C for chilled pharmaceuticals/perishables)
    - max_temp: Upper thermal threshold (e.g. 8.0°C for chilled pharmaceuticals/perishables)
    - max_allowed_excursion_minutes: Allowable cumulative excursion time before risk escalates
    """
    if df.empty or "temperature" not in df.columns:
        return {
            "risk_category": "UNKNOWN",
            "risk_score": 0,
            "reasons": ["No data provided for evaluation."],
            "recommendations": ["Upload or provide environmental dataset."],
            "metrics": {}
        }

    temp = df["temperature"].values
    n_points = len(df)
    
    # Excursion boolean masks
    high_excursion = temp > max_temp
    low_excursion = temp < min_temp
    any_excursion = high_excursion | low_excursion

    # Summary Statistics
    max_recorded = float(np.max(temp))
    min_recorded = float(np.min(temp))
    mean_temp = float(np.mean(temp))
    out_of_range_count = int(any_excursion.sum())
    percent_out_of_range = round((out_of_range_count / max(n_points, 1)) * 100.0, 2)

    # Estimate time duration
    total_duration_minutes = 0.0
    excursion_duration_minutes = 0.0
    if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        sorted_ts = df["timestamp"].sort_values()
        if len(sorted_ts) >= 2:
            time_delta = sorted_ts.iloc[-1] - sorted_ts.iloc[0]
            total_duration_minutes = time_delta.total_seconds() / 60.0
            # Excursion duration estimate proportional to out-of-range ratio
            excursion_duration_minutes = (percent_out_of_range / 100.0) * total_duration_minutes
    else:
        # Fallback assuming 30s per record
        total_duration_minutes = (n_points * 30.0) / 60.0
        excursion_duration_minutes = (out_of_range_count * 30.0) / 60.0

    excursion_duration_minutes = round(excursion_duration_minutes, 1)

    # Excursion Episodes Count (continuous blocks of excursions)
    # Detect transitions from False -> True
    transitions = np.diff(any_excursion.astype(int), prepend=0)
    num_excursion_episodes = int((transitions == 1).sum())

    # ML Anomaly frequency (if anomaly detection has been performed)
    has_ml_scores = "is_anomaly" in df.columns
    if has_ml_scores:
        anomaly_count = int(df["is_anomaly"].sum())
        anomaly_frequency_pct = round((anomaly_count / max(n_points, 1)) * 100.0, 2)
    else:
        anomaly_count = 0
        anomaly_frequency_pct = 0.0

    # Temperature Trend (°C / hour)
    trend_deg_per_hour = 0.0
    if n_points >= 5:
        x_steps = np.arange(n_points)
        slope, _ = np.polyfit(x_steps, temp, 1)
        # Convert slope per step to slope per hour
        step_minutes = (total_duration_minutes / max(n_points - 1, 1))
        steps_per_hour = 60.0 / max(step_minutes, 0.1)
        trend_deg_per_hour = round(float(slope * steps_per_hour), 2)

    # ==========================================
    # TRANSPARENT RULE-BASED SCORING
    # Total Score: 0 - 100
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

    # Freezing risk (very critical for vaccines, insulin, fresh produce)
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

    # Final Total Risk Score
    total_score = min(int(round(duration_pts + severity_pts + ml_pts + trend_pts)), 100)

    # Classify Category
    if total_score >= 75:
        category = "CRITICAL"
        recommendations = [
            "Immediately notify cold-chain operations and facility supervisor.",
            "Verify refrigeration compressor status and check for door seals or power disruption.",
            "Quarantine sensitive goods for quality assurance testing before release.",
            "Review thermal exposure log for regulatory compliance reporting."
        ]
    elif total_score >= 50:
        category = "HIGH"
        recommendations = [
            "Inspect cold storage unit and check physical door closure and loading activity.",
            "Verify thermostat settings and air circulation around sensor locations.",
            "Monitor product temperatures closely over the next operational shift.",
            "Prepare contingency cooling if temperature continues upward trend."
        ]
    elif total_score >= 25:
        category = "MODERATE"
        recommendations = [
            "Investigate brief excursion causes (e.g. routine door openings during restocking).",
            "Ensure refrigeration doors remain closed when not in active use.",
            "Observe subsequent readings to confirm return to standard operating range."
        ]
    else:
        category = "LOW"
        recommendations = [
            "Maintain standard monitoring routine.",
            "Cold-chain parameters remain well within acceptable operating band.",
            "No immediate operator action required."
        ]

    return {
        "risk_category": category,
        "risk_score": total_score,
        "point_breakdown": point_breakdown,
        "reasons": reasons,
        "recommendations": recommendations,
        "metrics": {
            "min_configured_temp": min_temp,
            "max_configured_temp": max_temp,
            "max_allowed_excursion_minutes": max_allowed_excursion_minutes,
            "max_recorded_temp": round(max_recorded, 2),
            "min_recorded_temp": round(min_recorded, 2),
            "mean_recorded_temp": round(mean_temp, 2),
            "num_excursion_episodes": num_excursion_episodes,
            "excursion_duration_minutes": excursion_duration_minutes,
            "percent_out_of_range": percent_out_of_range,
            "has_ml_scores": has_ml_scores,
            "anomaly_frequency_pct": anomaly_frequency_pct,
            "anomaly_count": anomaly_count,
            "trend_deg_per_hour": trend_deg_per_hour
        }
    }
