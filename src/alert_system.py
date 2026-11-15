"""
ColdGuard AI - Alert System Module
Detects immediate operational threshold violations, rapid thermal spikes,
consecutive anomalous observations, and ML score threshold crossings.
"""

from typing import List, Dict, Any, Optional
import pandas as pd


def generate_cold_chain_alerts(
    df: pd.DataFrame,
    min_temp: float = 2.0,
    max_temp: float = 8.0,
    rapid_change_threshold: float = 1.5,
    consecutive_anomaly_limit: int = 3
) -> pd.DataFrame:
    """
    Scans dataset and generates structured alerts.
    
    Alert triggers:
    1. Upper temperature exceedance (> max_temp)
    2. Lower temperature sub-cooling (< min_temp)
    3. Rapid temperature shift (|temperature_change| >= rapid_change_threshold)
    4. Consecutive abnormal observations (streak >= consecutive_anomaly_limit)
    5. ML anomaly score threshold breach (is_anomaly == True)
    """
    if df.empty or "temperature" not in df.columns:
        return pd.DataFrame(columns=[
            "timestamp", "sensor_id", "temperature", "humidity",
            "anomaly_score", "severity", "reason", "recommended_action"
        ])

    alerts: List[Dict[str, Any]] = []

    # Sort to evaluate consecutive streaks properly
    data = df.sort_values(by=["sensor_id", "timestamp"]).copy()

    for sensor_id, group in data.groupby("sensor_id"):
        consecutive_anomaly_streak = 0

        for _, row in group.iterrows():
            ts = row.get("timestamp", "N/A")
            temp = float(row.get("temperature", 0.0))
            hum = float(row.get("humidity", 0.0)) if "humidity" in row else None
            score = float(row.get("anomaly_score", 0.0)) if "anomaly_score" in row else None
            change = float(row.get("temperature_change", 0.0)) if "temperature_change" in row else 0.0
            is_ml_anomaly = bool(row.get("is_anomaly", False))

            # Track consecutive anomaly streak
            if is_ml_anomaly:
                consecutive_anomaly_streak += 1
            else:
                consecutive_anomaly_streak = 0

            # Trigger 1: Upper Temperature Excursion
            if temp > max_temp:
                severity = "CRITICAL" if temp > (max_temp + 5.0) else "WARNING"
                alerts.append({
                    "timestamp": ts,
                    "sensor_id": sensor_id,
                    "temperature": round(temp, 2),
                    "humidity": round(hum, 1) if hum is not None else None,
                    "anomaly_score": round(score, 2) if score is not None else None,
                    "severity": severity,
                    "alert_type": "High Temperature Excursion",
                    "reason": f"Temperature {temp:.1f}°C exceeds safe maximum threshold of {max_temp:.1f}°C.",
                    "recommended_action": "Check door seal and verify primary chiller cooling cycle."
                })

            # Trigger 2: Lower Temperature Excursion (Freezing risk)
            elif temp < min_temp:
                severity = "CRITICAL" if temp < 0.0 else "WARNING"
                alerts.append({
                    "timestamp": ts,
                    "sensor_id": sensor_id,
                    "temperature": round(temp, 2),
                    "humidity": round(hum, 1) if hum is not None else None,
                    "anomaly_score": round(score, 2) if score is not None else None,
                    "severity": severity,
                    "alert_type": "Low Temperature Excursion",
                    "reason": f"Temperature {temp:.1f}°C fell below safe minimum threshold of {min_temp:.1f}°C.",
                    "recommended_action": "Prevent product freeze damage; check thermostat and defrost heater."
                })

            # Trigger 3: Rapid Temperature Shift
            if abs(change) >= rapid_change_threshold:
                direction = "rise" if change > 0 else "drop"
                alerts.append({
                    "timestamp": ts,
                    "sensor_id": sensor_id,
                    "temperature": round(temp, 2),
                    "humidity": round(hum, 1) if hum is not None else None,
                    "anomaly_score": round(score, 2) if score is not None else None,
                    "severity": "WARNING",
                    "alert_type": "Rapid Temperature Shift",
                    "reason": f"Unusual sudden temperature {direction} of {abs(change):.2f}°C in a single step.",
                    "recommended_action": "Check for open access doors or loading of unconditioned stock."
                })

            # Trigger 4: Consecutive Anomaly Streak
            if consecutive_anomaly_streak >= consecutive_anomaly_limit:
                alerts.append({
                    "timestamp": ts,
                    "sensor_id": sensor_id,
                    "temperature": round(temp, 2),
                    "humidity": round(hum, 1) if hum is not None else None,
                    "anomaly_score": round(score, 2) if score is not None else None,
                    "severity": "CRITICAL",
                    "alert_type": "Persistent Anomaly Pattern",
                    "reason": f"Sustained abnormal environmental pattern: {consecutive_anomaly_streak} consecutive anomalies detected.",
                    "recommended_action": "Perform immediate on-site inspection; sustained failure risk."
                })

            # Trigger 5: Anomaly Score Crossing Threshold (when within normal temp bounds but pattern is abnormal)
            elif is_ml_anomaly and (min_temp <= temp <= max_temp):
                alerts.append({
                    "timestamp": ts,
                    "sensor_id": sensor_id,
                    "temperature": round(temp, 2),
                    "humidity": round(hum, 1) if hum is not None else None,
                    "anomaly_score": round(score, 2) if score is not None else None,
                    "severity": "INFO",
                    "alert_type": "ML Pattern Anomaly",
                    "reason": f"ML anomaly score ({score:.1f}) crossed threshold despite temperature being within nominal range.",
                    "recommended_action": "Monitor sensor trend for early signs of compressor cycle degradation."
                })

    alert_df = pd.DataFrame(alerts)
    if alert_df.empty:
        return pd.DataFrame(columns=[
            "timestamp", "sensor_id", "temperature", "humidity",
            "anomaly_score", "severity", "alert_type", "reason", "recommended_action"
        ])

    # De-duplicate identical timestamp + sensor_id + alert_type
    alert_df = alert_df.drop_duplicates(subset=["timestamp", "sensor_id", "alert_type"]).reset_index(drop=True)
    return alert_df
