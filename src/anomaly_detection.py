"""
ColdGuard AI - Anomaly Detection Engine
Implements unsupervised anomaly detection for cold-storage monitoring.
Supports:
- Local Demo Mode: Scikit-Learn Isolation Forest & tree-depth scoring running locally on CPU.
- Score normalization, distribution statistics (mean, std), and configurable 3-sigma thresholds.
- Plain-English interpretation of why an observation was flagged as an anomaly.
"""

from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


FEATURE_COLS = [
    "temperature",
    "humidity",
    "temperature_change",
    "temperature_deviation"
]


class LocalAnomalyDetector:
    """
    Genuine local machine learning model for cold-chain anomaly detection.
    Uses an Isolation Forest ensemble to detect multi-dimensional anomalies
    in temperature, humidity, rate-of-change, and rolling deviations.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.05,
        random_state: int = 42
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.is_fitted = False
        self.score_mean: float = 0.0
        self.score_std: float = 1.0

    def fit_predict(
        self,
        df: pd.DataFrame,
        sigma_multiplier: float = 3.0,
        features: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Trains the local model on the dataset and scores each observation.
        
        Score Transformation:
        - Scikit-learn's IsolationForest score_samples() outputs negative values (lower = more anomalous).
        - We invert and rescale to a standard scale where:
          - Higher score = More anomalous
          - Baseline normal scores hover around 10-30
          - Severe outliers cross 60-100+
        """
        if df.empty:
            return df, {}

        feature_cols = features or [c for c in FEATURE_COLS if c in df.columns]
        if not feature_cols:
            raise ValueError(f"None of the required features {FEATURE_COLS} were found in data.")

        # Prepare feature matrix, filling any remaining NaNs with 0
        X = df[feature_cols].copy().fillna(0.0).values

        # Initialize and fit Isolation Forest
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        self.model.fit(X)
        self.is_fitted = True

        # Compute raw scores: score_samples returns opposite of anomaly score
        raw_scores = self.model.score_samples(X)
        
        # Invert so higher = more anomalous: inverted_score = -raw_scores
        # Typical raw_scores range from -0.8 (extreme outlier) to -0.35 (inlier)
        # Normalize to 0-100 scale:
        inverted = -raw_scores
        min_val = np.percentile(inverted, 1)
        max_val = np.percentile(inverted, 99)
        val_range = max(max_val - min_val, 1e-5)
        
        # Scale to 0-100 base
        scaled_scores = ((inverted - min_val) / val_range) * 50.0 + 10.0
        scaled_scores = np.clip(scaled_scores, 0.0, 150.0)

        result_df = df.copy()
        result_df["anomaly_score"] = np.round(scaled_scores, 2)

        # Statistical thresholding: mean + k * std
        self.score_mean = float(np.mean(scaled_scores))
        self.score_std = float(np.std(scaled_scores))
        computed_threshold = round(self.score_mean + (sigma_multiplier * self.score_std), 2)

        result_df["anomaly_threshold"] = computed_threshold
        result_df["is_anomaly"] = result_df["anomaly_score"] >= computed_threshold
        result_df["model_source"] = "LOCAL (Scikit-Learn IsolationForest)"

        # Classify severity
        def _classify_severity(row):
            score = row["anomaly_score"]
            thresh = row["anomaly_threshold"]
            if score < thresh:
                return "Normal"
            elif score < (thresh + self.score_std):
                return "Moderate Anomaly"
            elif score < (thresh + 2 * self.score_std):
                return "High Anomaly"
            else:
                return "Critical Anomaly"

        result_df["anomaly_severity"] = result_df.apply(_classify_severity, axis=1)

        # Add plain-English explanations for anomalies
        result_df["anomaly_reason"] = [
            explain_anomaly(row, self.score_mean, computed_threshold)
            if row["is_anomaly"] else "Normal environmental pattern"
            for _, row in result_df.iterrows()
        ]

        summary = {
            "model_type": "Local Isolation Forest (Unsupervised)",
            "execution_mode": "LOCAL",
            "features_used": feature_cols,
            "observations_evaluated": len(result_df),
            "score_mean": round(self.score_mean, 2),
            "score_std": round(self.score_std, 2),
            "threshold_applied": computed_threshold,
            "sigma_multiplier": sigma_multiplier,
            "anomalies_detected": int(result_df["is_anomaly"].sum()),
            "anomaly_rate_percent": round(float((result_df["is_anomaly"].mean()) * 100), 2)
        }

        return result_df, summary


def explain_anomaly(
    row: pd.Series,
    score_mean: float,
    threshold: float
) -> str:
    """
    Generates a beginner-friendly explanation of why an observation was flagged.
    """
    temp = row.get("temperature", 0.0)
    change = row.get("temperature_change", 0.0)
    dev = row.get("temperature_deviation", 0.0)
    hum = row.get("humidity", 0.0)
    score = row.get("anomaly_score", 0.0)

    reasons: List[str] = []

    if abs(change) >= 1.0:
        reasons.append(f"Rapid temperature shift ({change:+.2f}°C/step)")
    if abs(dev) >= 2.0:
        reasons.append(f"Strong deviation from baseline ({dev:+.2f}°C)")
    if temp > 10.0:
        reasons.append(f"Elevated temperature reading ({temp:.1f}°C)")
    elif temp < 1.0:
        reasons.append(f"Freezing temperature reading ({temp:.1f}°C)")
    if hum > 90.0:
        reasons.append(f"Extreme relative humidity ({hum:.1f}%)")
    elif hum < 40.0:
        reasons.append(f"Dry air reading ({hum:.1f}%)")

    if not reasons:
        reasons.append(f"Multi-feature statistical anomaly (Score: {score:.1f} vs Thresh: {threshold:.1f})")

    return "; ".join(reasons)
