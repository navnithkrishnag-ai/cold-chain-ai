"""
ColdGuard AI - Amazon SageMaker AI Random Cut Forest Service
Coordinates model training, batch transform inference, and score retrieval
using Amazon SageMaker AI's built-in Random Cut Forest algorithm.
"""

import os
import json
import time
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import boto3
from botocore.exceptions import ClientError


# SageMaker ECR built-in algorithm container image mapping for verified regions
RCF_IMAGE_URIS = {
    "ap-south-1": "991648021394.dkr.ecr.ap-south-1.amazonaws.com/randomcutforest:latest",
    "us-east-1": "382416823181.dkr.ecr.us-east-1.amazonaws.com/randomcutforest:latest",
    "us-east-2": "433757028032.dkr.ecr.us-east-2.amazonaws.com/randomcutforest:latest",
    "us-west-2": "174872318102.dkr.ecr.us-west-2.amazonaws.com/randomcutforest:latest",
    "eu-west-1": "438346465227.dkr.ecr.eu-west-1.amazonaws.com/randomcutforest:latest",
    "eu-central-1": "664544806723.dkr.ecr.eu-central-1.amazonaws.com/randomcutforest:latest"
}

RCF_FEATURES = [
    "temperature",
    "humidity",
    "temperature_change",
    "temperature_deviation"
]


def parse_rcf_jsonlines_scores(scores_text: str, expected_count: int) -> List[float]:
    """
    Parses SageMaker RCF Batch Transform JSON Lines output.
    Format per line: {"score": <float>}
    
    Validation:
    - Every input record must have a corresponding valid score.
    - Missing or invalid scores are NOT silently replaced with zero.
    - A count mismatch raises a clear ValueError.
    """
    if not scores_text or not scores_text.strip():
        raise ValueError(f"SageMaker RCF output is empty. Expected {expected_count} scores.")

    scores_list: List[float] = []
    lines = scores_text.strip().splitlines()

    for idx, line in enumerate(lines):
        trimmed = line.strip()
        if not trimmed:
            continue
        try:
            parsed = json.loads(trimmed)
            if not isinstance(parsed, dict) or "score" not in parsed:
                raise ValueError(f"Record {idx + 1} missing 'score' key: '{trimmed}'")
            score_val = parsed["score"]
            if score_val is None or not isinstance(score_val, (int, float)) or np.isnan(score_val):
                raise ValueError(f"Record {idx + 1} contains non-numeric score: '{trimmed}'")
            scores_list.append(float(score_val))
        except json.JSONDecodeError:
            # Check if line is raw float
            try:
                score_val = float(trimmed)
                if np.isnan(score_val):
                    raise ValueError
                scores_list.append(score_val)
            except ValueError:
                raise ValueError(f"Malformed JSON Lines output at record {idx + 1}: '{trimmed}'")

    if len(scores_list) != expected_count:
        raise ValueError(
            f"Score count mismatch: expected {expected_count} scores, but received {len(scores_list)} from SageMaker RCF output."
        )

    return scores_list


def split_chronological_train_eval(
    df: pd.DataFrame,
    train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits multi-sensor time series chronologically into:
    1. df_train: Historical baseline observations (first train_ratio of timestamps).
    2. df_eval: Evaluation observations to monitor and score for anomalies.
    
    Preserves all columns, timestamps, and sensor IDs.
    """
    if df.empty:
        return df.copy(), df.copy()

    # Sort chronologically by timestamp
    sorted_df = df.sort_values(by=["timestamp"]).reset_index(drop=True)
    split_idx = max(int(len(sorted_df) * train_ratio), 1)

    df_train = sorted_df.iloc[:split_idx].copy().reset_index(drop=True)
    df_eval = sorted_df.iloc[split_idx:].copy().reset_index(drop=True)

    # Fallback if evaluation split is empty (e.g. very small dataset)
    if df_eval.empty:
        df_eval = df_train.copy()

    return df_train, df_eval


class SageMakerRCFService:
    """
    Integrates with Amazon SageMaker AI Random Cut Forest (RCF).
    Uses cost-effective Batch Transform to eliminate expensive 24/7 endpoint charges.
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        role_arn: Optional[str] = None,
        region_name: str = "ap-south-1"
    ):
        self.bucket_name = bucket_name or os.getenv("COLDGUARD_S3_BUCKET", "")
        self.role_arn = role_arn or os.getenv("COLDGUARD_SAGEMAKER_ROLE_ARN", "")
        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
        self._sm_client = None
        self.scaler: Optional[StandardScaler] = None

    def get_client(self):
        """Initializes and returns boto3 sagemaker client."""
        if self._sm_client is None:
            self._sm_client = boto3.client("sagemaker", region_name=self.region_name)
        return self._sm_client

    def get_rcf_image_uri(self) -> str:
        """
        Returns the verified ECR image URI for Random Cut Forest in the configured region.
        Raises ValueError for unsupported regions to avoid invalid registry IDs.
        """
        # 1. Check verified mapping
        if self.region_name in RCF_IMAGE_URIS:
            return RCF_IMAGE_URIS[self.region_name]

        # 2. Try AWS SDK resolver if available
        try:
            import sagemaker
            return sagemaker.image_uris.retrieve("randomcutforest", self.region_name)
        except Exception:
            pass

        # 3. Raise clear error for unsupported regions
        supported = ", ".join(sorted(RCF_IMAGE_URIS.keys()))
        raise ValueError(
            f"Unsupported AWS region for SageMaker Random Cut Forest: '{self.region_name}'. "
            f"Supported verified regions are: {supported}."
        )

    def validate_cloud_prerequisites(self, s3_manager=None) -> Dict[str, Any]:
        """
        Validates whether necessary AWS resources and IAM role are present.
        Uses S3Manager.check_connection() to verify:
        - AWS authentication via STS.
        - Configured S3 bucket exists and is accessible.
        - SageMaker API is accessible.
        - Execution Role ARN is configured.
        """
        issues: List[str] = []
        instructions: List[str] = []

        # 1. Execution Role ARN check
        if not self.role_arn or not self.role_arn.strip():
            issues.append("COLDGUARD_SAGEMAKER_ROLE_ARN is not configured.")
            instructions.append(
                "Create an IAM Execution Role with 'AmazonSageMakerFullAccess' and S3 access, then set its ARN in .env."
            )

        # 2. S3 & STS verification using S3Manager
        if s3_manager is None:
            from src.aws_s3 import S3Manager
            s3_manager = S3Manager(bucket_name=self.bucket_name, region_name=self.region_name)

        s3_status = s3_manager.check_connection()
        if not s3_status["has_credentials"]:
            issues.append(s3_status.get("error_message") or "AWS credentials not found.")
            instructions.extend(s3_status.get("instructions", []))
        elif not s3_status["bucket_accessible"]:
            issues.append(s3_status.get("error_message") or f"S3 bucket '{self.bucket_name}' not accessible.")
            instructions.extend(s3_status.get("instructions", []))

        # 3. SageMaker API accessibility check
        can_reach_sm = False
        if s3_status["has_credentials"]:
            try:
                sm = self.get_client()
                sm.list_training_jobs(MaxResults=1)
                can_reach_sm = True
            except Exception as e:
                issues.append(f"SageMaker API connection check failed: {str(e)}")
                instructions.append("Verify your IAM permissions include 'sagemaker:ListTrainingJobs'.")

        is_ready = (
            len(issues) == 0
            and s3_status["has_credentials"]
            and s3_status["bucket_accessible"]
            and bool(self.role_arn and self.role_arn.strip())
            and can_reach_sm
        )

        return {
            "ready": is_ready,
            "has_credentials": s3_status["has_credentials"],
            "bucket_accessible": s3_status["bucket_accessible"],
            "bucket": self.bucket_name,
            "role_arn": self.role_arn,
            "region": self.region_name,
            "caller_arn": s3_status.get("caller_arn"),
            "issues": issues,
            "instructions": instructions
        }

    def prepare_scaled_features(
        self,
        df: pd.DataFrame,
        is_training: bool = False
    ) -> pd.DataFrame:
        """
        Prepares and standardizes numeric features for Random Cut Forest.
        Features: temperature, humidity, temperature_change, temperature_deviation.
        
        Standardization:
        - Prevents high-variance features (e.g. humidity 20-100) from dominating low-variance features (e.g. temperature change).
        - Fits StandardScaler on training data only; reuses fitted scaler for inference.
        """
        missing_features = [f for f in RCF_FEATURES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required numeric features for SageMaker RCF: {missing_features}")

        raw_features = df[RCF_FEATURES].copy().fillna(0.0).values

        if is_training or self.scaler is None:
            self.scaler = StandardScaler()
            scaled_features = self.scaler.fit_transform(raw_features)
        else:
            scaled_features = self.scaler.transform(raw_features)

        return pd.DataFrame(scaled_features, columns=RCF_FEATURES)

    def build_training_job_params(
        self,
        job_name: str,
        train_s3_uri: str,
        output_s3_uri: str,
        job_timeout_seconds: int = 600
    ) -> Dict[str, Any]:
        """
        Constructs the CreateTrainingJob parameter dictionary.
        Uses ShardedByS3Key for the RCF training channel.
        """
        image_uri = self.get_rcf_image_uri()
        return {
            "TrainingJobName": job_name,
            "AlgorithmSpecification": {
                "TrainingImage": image_uri,
                "TrainingInputMode": "File"
            },
            "RoleArn": self.role_arn,
            "InputDataConfig": [
                {
                    "ChannelName": "train",
                    "DataSource": {
                        "S3DataSource": {
                            "S3DataType": "S3Prefix",
                            "S3Uri": train_s3_uri,
                            "S3DataDistributionType": "ShardedByS3Key"
                        }
                    },
                    "ContentType": "text/csv;label_size=0"
                }
            ],
            "OutputDataConfig": {
                "S3OutputPath": output_s3_uri
            },
            "ResourceConfig": {
                "InstanceType": "ml.m5.large",
                "InstanceCount": 1,
                "VolumeSizeInGB": 10
            },
            "StoppingCondition": {
                "MaxRuntimeInSeconds": job_timeout_seconds
            },
            "HyperParameters": {
                "feature_dim": str(len(RCF_FEATURES)),
                "num_trees": "100",
                "num_samples_per_tree": "256"
            }
        }

    def build_transform_job_params(
        self,
        transform_job_name: str,
        model_name: str,
        eval_s3_uri: str,
        transform_output_s3: str
    ) -> Dict[str, Any]:
        """
        Constructs the CreateTransformJob parameter dictionary.
        Explicitly sets Accept = application/jsonlines.
        """
        return {
            "TransformJobName": transform_job_name,
            "ModelName": model_name,
            "TransformInput": {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": eval_s3_uri
                    }
                },
                "ContentType": "text/csv;label_size=0",
                "SplitType": "Line"
            },
            "TransformOutput": {
                "S3OutputPath": transform_output_s3,
                "Accept": "application/jsonlines",
                "AssembleWith": "Line"
            },
            "TransformResources": {
                "InstanceType": "ml.m5.large",
                "InstanceCount": 1
            }
        }

    def run_rcf_workflow(
        self,
        df: pd.DataFrame,
        s3_manager,
        sigma_multiplier: float = 3.0,
        job_timeout_seconds: int = 600,
        train_ratio: float = 0.7
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes genuine SageMaker RCF Workflow:
        1. Chronological training/inference split (baseline vs evaluation period).
        2. Fits StandardScaler on training data; transforms inference data.
        3. Uploads training data to S3.
        4. Submits SageMaker Training Job for RCF with ShardedByS3Key.
        5. Creates SageMaker Model.
        6. Submits Batch Transform Job with Accept = application/jsonlines.
        7. Downloads scores from S3 and parses with strict record count validation.
        8. Preserves timestamps, sensor IDs, and all original attributes.
        """
        prereqs = self.validate_cloud_prerequisites(s3_manager)
        if not prereqs["ready"]:
            raise RuntimeError(
                f"Cannot execute AWS Cloud Mode. Issues: {'; '.join(prereqs['issues'])}"
            )

        sm = self.get_client()
        timestamp_id = int(time.time())
        job_name = f"coldguard-rcf-{timestamp_id}"
        model_name = f"coldguard-rcf-model-{timestamp_id}"
        transform_job_name = f"coldguard-rcf-transform-{timestamp_id}"

        # 1. Chronological Train/Inference Split
        df_train, df_eval = split_chronological_train_eval(df, train_ratio=train_ratio)

        # 2. Consistent Feature Scaling (Fit on Train, Transform on Eval)
        X_train_scaled = self.prepare_scaled_features(df_train, is_training=True)
        X_eval_scaled = self.prepare_scaled_features(df_eval, is_training=False)

        # 3. Upload to S3 without headers (required by SageMaker RCF)
        train_s3_key = f"coldguard/training_input/{job_name}.csv"
        success, train_s3_uri = s3_manager.upload_dataframe_as_csv(X_train_scaled, train_s3_key, include_header=False)
        if not success:
            raise RuntimeError(f"Failed to upload training data to S3: {train_s3_uri}")

        eval_s3_key = f"coldguard/inference_input/{transform_job_name}.csv"
        success_eval, eval_s3_uri = s3_manager.upload_dataframe_as_csv(X_eval_scaled, eval_s3_key, include_header=False)
        if not success_eval:
            raise RuntimeError(f"Failed to upload inference data to S3: {eval_s3_uri}")

        output_s3_uri = f"s3://{self.bucket_name}/coldguard/models/{job_name}/"
        image_uri = self.get_rcf_image_uri()

        # 4. Create SageMaker Training Job with ShardedByS3Key
        training_params = self.build_training_job_params(
            job_name=job_name,
            train_s3_uri=train_s3_uri,
            output_s3_uri=output_s3_uri,
            job_timeout_seconds=job_timeout_seconds
        )
        sm.create_training_job(**training_params)

        # Wait for Training Job completion
        start_time = time.time()
        while True:
            desc = sm.describe_training_job(TrainingJobName=job_name)
            status = desc["TrainingJobStatus"]
            if status == "Completed":
                model_artifact_s3 = desc["ModelArtifacts"]["S3ModelArtifacts"]
                break
            elif status in ["Failed", "Stopped"]:
                failure_reason = desc.get("FailureReason", "Unknown failure")
                raise RuntimeError(f"SageMaker Training Job failed: {failure_reason}")

            if (time.time() - start_time) > job_timeout_seconds:
                raise TimeoutError("SageMaker Training Job timed out.")
            time.sleep(10)

        # 5. Create SageMaker Model from artifact
        model_params = {
            "ModelName": model_name,
            "PrimaryContainer": {
                "Image": image_uri,
                "ModelDataUrl": model_artifact_s3
            },
            "ExecutionRoleArn": self.role_arn
        }
        sm.create_model(**model_params)

        # 6. Create Batch Transform Job with Accept = application/jsonlines
        transform_output_prefix = f"coldguard/batch_inference_output/{transform_job_name}/"
        transform_output_s3 = f"s3://{self.bucket_name}/{transform_output_prefix}"

        transform_params = self.build_transform_job_params(
            transform_job_name=transform_job_name,
            model_name=model_name,
            eval_s3_uri=eval_s3_uri,
            transform_output_s3=transform_output_s3
        )
        sm.create_transform_job(**transform_params)

        # Wait for Batch Transform completion
        start_time = time.time()
        while True:
            desc_trans = sm.describe_transform_job(TransformJobName=transform_job_name)
            status = desc_trans["TransformJobStatus"]
            if status == "Completed":
                break
            elif status in ["Failed", "Stopped"]:
                failure_reason = desc_trans.get("FailureReason", "Unknown transform failure")
                raise RuntimeError(f"SageMaker Batch Transform Job failed: {failure_reason}")

            if (time.time() - start_time) > job_timeout_seconds:
                raise TimeoutError("SageMaker Batch Transform Job timed out.")
            time.sleep(10)

        # 7. Retrieve scores from S3 and parse JSON Lines output
        output_file_key = f"{transform_output_prefix}{transform_job_name}.csv.out"
        success, scores_text = s3_manager.download_s3_file_as_string(output_file_key)
        if not success:
            raise RuntimeError(f"Could not retrieve anomaly scores from S3: {scores_text}")

        # Parse JSON Lines scores with validation
        scores_list = parse_rcf_jsonlines_scores(scores_text, expected_count=len(df_eval))

        # 8. Merge scores preserving timestamp, sensor_id, and metadata
        result_df = df_eval.copy()
        raw_scores = np.array(scores_list)
        result_df["anomaly_score"] = np.round(raw_scores, 2)

        score_mean = float(np.mean(raw_scores))
        score_std = float(np.std(raw_scores))
        threshold = round(score_mean + (sigma_multiplier * score_std), 2)

        result_df["anomaly_threshold"] = threshold
        result_df["is_anomaly"] = result_df["anomaly_score"] >= threshold
        result_df["model_source"] = "AWS SageMaker AI (Random Cut Forest)"

        summary = {
            "execution_mode": "AWS",
            "model_type": "Amazon SageMaker AI Random Cut Forest",
            "training_job_name": job_name,
            "transform_job_name": transform_job_name,
            "train_observations": len(df_train),
            "eval_observations": len(df_eval),
            "observations_evaluated": len(result_df),
            "score_mean": round(score_mean, 2),
            "score_std": round(score_std, 2),
            "threshold_applied": threshold,
            "anomalies_detected": int(result_df["is_anomaly"].sum()),
            "anomaly_rate_percent": round(float(result_df["is_anomaly"].mean() * 100), 2)
        }

        return result_df, summary
