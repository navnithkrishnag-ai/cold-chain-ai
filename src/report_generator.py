"""
ColdGuard AI - Report Generator Module
Produces comprehensive cold-chain audit summaries, anomaly logs,
and sensor-wise statistical reports for export.
"""

from datetime import datetime
from typing import Dict, Any
import pandas as pd


def generate_sensor_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes sensor-wise operational summary metrics.
    """
    if df.empty or "sensor_id" not in df.columns:
        return pd.DataFrame()

    def _sensor_stats(g):
        temp = g["temperature"]
        hum = g["humidity"] if "humidity" in g else pd.Series([0.0])
        anomalies = g["is_anomaly"].sum() if "is_anomaly" in g else 0
        total_obs = len(g)
        
        return pd.Series({
            "Observations": total_obs,
            "Temp Min (°C)": round(temp.min(), 1),
            "Temp Max (°C)": round(temp.max(), 1),
            "Temp Mean (°C)": round(temp.mean(), 1),
            "Temp Std (°C)": round(temp.std(), 2) if total_obs > 1 else 0.0,
            "Hum Mean (%)": round(hum.mean(), 1),
            "Anomalies": int(anomalies),
            "Anomaly Rate (%)": round((anomalies / max(total_obs, 1)) * 100, 1)
        })

    summary = df.groupby("sensor_id").apply(_sensor_stats, include_groups=False).reset_index()
    return summary


def generate_audit_report_text(
    risk_data: Dict[str, Any],
    anomaly_summary: Dict[str, Any],
    df: pd.DataFrame
) -> str:
    """
    Generates a formal plain-text / Markdown cold-chain condition audit report.
    """
    metrics = risk_data.get("metrics", {})
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    report_lines = [
        "=" * 70,
        "COLDGUARD AI - COLD-CHAIN CONDITION & RISK AUDIT REPORT",
        "=" * 70,
        f"Generated At          : {now_str}",
        f"Execution Mode        : {anomaly_summary.get('execution_mode', 'LOCAL')}",
        f"ML Model Algorithm    : {anomaly_summary.get('model_type', 'Unsupervised Anomaly Detection')}",
        f"Total Records Analyzed: {len(df)}",
        f"Monitored Sensors     : {', '.join(df['sensor_id'].unique()) if 'sensor_id' in df else 'N/A'}",
        "-" * 70,
        "1. EXECUTIVE RISK ASSESSMENT",
        "-" * 70,
        f"Overall Risk Category : {risk_data.get('risk_category', 'UNKNOWN')}",
        f"Cold-Chain Risk Score : {risk_data.get('risk_score', 0)} / 100",
        "",
        "Risk Classification Factors:",
    ]

    for reason in risk_data.get("reasons", []):
        report_lines.append(f"  • {reason}")

    report_lines.extend([
        "",
        "Recommended Operator Actions:",
    ])
    for rec in risk_data.get("recommendations", []):
        report_lines.append(f"  [!] {rec}")

    report_lines.extend([
        "",
        "-" * 70,
        "2. THERMAL EXCURSION METRICS",
        "-" * 70,
        f"Operating Range Preset : {metrics.get('min_configured_temp', 2.0)}°C to {metrics.get('max_configured_temp', 8.0)}°C",
        f"Max Allowed Excursion  : {metrics.get('max_allowed_excursion_minutes', 30.0)} minutes",
        f"Max Temperature Logged : {metrics.get('max_recorded_temp', 'N/A')}°C",
        f"Min Temperature Logged : {metrics.get('min_recorded_temp', 'N/A')}°C",
        f"Mean Temperature Logged: {metrics.get('mean_recorded_temp', 'N/A')}°C",
        f"Excursion Episodes     : {metrics.get('num_excursion_episodes', 0)}",
        f"Est. Excursion Duration: {metrics.get('excursion_duration_minutes', 0.0)} minutes",
        f"Percent Out of Range   : {metrics.get('percent_out_of_range', 0.0)}%",
        f"Temperature Trend      : {metrics.get('trend_deg_per_hour', 0.0)}°C/hour",
        "",
        "-" * 70,
        "3. MACHINE LEARNING ANOMALY FINDINGS",
        "-" * 70,
        f"Model Source           : {df['model_source'].iloc[0] if 'model_source' in df else 'Local'}",
        f"Total Anomalies Flagged: {anomaly_summary.get('anomalies_detected', 0)}",
        f"Anomaly Rate           : {anomaly_summary.get('anomaly_rate_percent', 0.0)}%",
        f"Threshold Applied      : {anomaly_summary.get('threshold_applied', 'N/A')}",
        f"Baseline Score Mean    : {anomaly_summary.get('score_mean', 'N/A')}",
        f"Baseline Score Std     : {anomaly_summary.get('score_std', 'N/A')}",
        "",
        "=" * 70,
        "DISCLAIMER: ColdGuard AI provides condition monitoring and early warning alerts.",
        "It does not claim to predict mechanical failures or spoilage with certainty.",
        "=" * 70
    ])

    return "\n".join(report_lines)
