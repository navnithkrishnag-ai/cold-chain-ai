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
import boto3
from botocore.exceptions import ClientError


# SageMaker ECR built-in algorithm container image mapping by region
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

    def get_client(self):
        """Initializes and returns boto3 sagemaker client."""
        if self._sm_client is None:
            self._sm_client = boto3.client("sagemaker", region_name=self.region_name)
        return self._sm_client

    def get_rcf_image_uri(self) -> str:
        """Returns the ECR image URI for Random Cut Forest in the configured region."""
        return RCF_IMAGE_URIS.get(
            self.region_name,
            f"991648021394.dkr.ecr.{self.region_name}.amazonaws.com/randomcutforest:latest"
        )

    def validate_cloud_prerequisites(self) -> Dict[str, Any]:
        """
        Validates whether necessary AWS resources and IAM role are present.
        """
        issues: List[str] = []
        instructions: List[str] = []

        if not self.bucket_name:
            issues.append("COLDGUARD_S3_BUCKET is not set.")
            instructions.append("Create an S3 bucket in your AWS account and set COLDGUARD_S3_BUCKET in .env.")

        if not self.role_arn:
            issues.append("COLDGUARD_SAGEMAKER_ROLE_ARN is not set.")
            instructions.append(
                "Create an IAM Execution Role with 'AmazonSageMakerFullAccess' and S3 access, then set its ARN in .env."
            )

        try:
            # Check if SageMaker API is reachable
            sm = self.get_client()
            sm.list_training_jobs(MaxResults=1)
            can_reach_sm = True
        except Exception as e:
            can_reach_sm = False
            issues.append(f"SageMaker API connection check failed: {str(e)}")
            instructions.append("Verify AWS credentials, internet connectivity, and IAM permissions.")

        return {
            "ready": len(issues) == 0 and can_reach_sm,
            "bucket": self.bucket_name,
            "role_arn": self.role_arn,
            "region": self.region_name,
            "issues": issues,
            "instructions": instructions
        }

    def prepare_rcf_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extracts numeric training features required by SageMaker RCF.
        SageMaker RCF expects header-less CSV with float values.
        Returns (feature_matrix_df, full_df).
        """
        missing_features = [f for f in RCF_FEATURES if f not in df.columns]
        if missing_features:
            raise ValueError(f"Missing required numeric features for SageMaker RCF: {missing_features}")

        # Extract only numeric features for SageMaker
        X_df = df[RCF_FEATURES].copy().fillna(0.0)
        return X_df, df

    def run_rcf_workflow(
        self,
        df: pd.DataFrame,
        s3_manager,
        sigma_multiplier: float = 3.0,
        job_timeout_seconds: int = 600
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Executes genuine SageMaker RCF Workflow:
        1. Prepares numeric CSV without headers
        2. Uploads data to S3
        3. Submits SageMaker Training Job for RCF
        4. Submits SageMaker Model creation
        5. Executes Batch Transform Job for inference
        6. Downloads scores from S3 and merges with original DataFrame
        """
        prereqs = self.validate_cloud_prerequisites()
        if not prereqs["ready"]:
            raise RuntimeError(
                f"Cannot execute AWS Cloud Mode. Issues: {'; '.join(prereqs['issues'])}"
            )

        sm = self.get_client()
        timestamp_id = int(time.time())
        job_name = f"coldguard-rcf-{timestamp_id}"
        model_name = f"coldguard-rcf-model-{timestamp_id}"
        transform_job_name = f"coldguard-rcf-transform-{timestamp_id}"

        # 1. Prepare numeric features
        X_df, original_df = self.prepare_rcf_dataset(df)

        # 2. Upload to S3 without header (required by SageMaker RCF)
        train_s3_key = f"coldguard/training_input/{job_name}.csv"
        success, train_s3_uri = s3_manager.upload_dataframe_as_csv(X_df, train_s3_key, include_header=False)
        if not success:
            raise RuntimeError(f"Failed to upload training data to S3: {train_s3_uri}")

        output_s3_uri = f"s3://{self.bucket_name}/coldguard/models/{job_name}/"
        image_uri = self.get_rcf_image_uri()

        # 3. Create SageMaker Training Job
        training_params = {
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
                            "S3DataDistributionType": "FullyReplicated"
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

        # 4. Create SageMaker Model from artifact
        model_params = {
            "ModelName": model_name,
            "PrimaryContainer": {
                "Image": image_uri,
                "ModelDataUrl": model_artifact_s3
            },
            "ExecutionRoleArn": self.role_arn
        }
        sm.create_model(**model_params)

        # 5. Create Batch Transform Job
        transform_output_prefix = f"coldguard/batch_inference_output/{transform_job_name}/"
        transform_output_s3 = f"s3://{self.bucket_name}/{transform_output_prefix}"

        transform_params = {
            "TransformJobName": transform_job_name,
            "ModelName": model_name,
            "TransformInput": {
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": train_s3_uri
                    }
                },
                "ContentType": "text/csv;label_size=0",
                "SplitType": "Line"
            },
            "TransformOutput": {
                "S3OutputPath": transform_output_s3,
                "AssembleWith": "Line"
            },
            "TransformResources": {
                "InstanceType": "ml.m5.large",
                "InstanceCount": 1
            }
        }
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

        # 6. Retrieve scores from S3
        output_file_key = f"{transform_output_prefix}{job_name}.csv.out"
        success, scores_text = s3_manager.download_s3_file_as_string(output_file_key)
        if not success:
            raise RuntimeError(f"Could not retrieve anomaly scores from S3: {scores_text}")

        # Parse scores JSON lines or CSV
        scores_list = []
        for line in scores_text.strip().splitlines():
            if not line:
                continue
            try:
                # SageMaker RCF output format: {"score": 1.234}
                parsed = json.loads(line)
                scores_list.append(float(parsed.get("score", 0.0)))
            except Exception:
                # Fallback if raw float
                scores_list.append(float(line.strip()))

        if len(scores_list) != len(original_df):
            raise ValueError(f"Score count ({len(scores_list)}) does not match input row count ({len(original_df)}).")

        # Merge results into dataframe
        result_df = original_df.copy()
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
            "observations_evaluated": len(result_df),
            "score_mean": round(score_mean, 2),
            "score_std": round(score_std, 2),
            "threshold_applied": threshold,
            "anomalies_detected": int(result_df["is_anomaly"].sum()),
            "anomaly_rate_percent": round(float(result_df["is_anomaly"].mean() * 100), 2)
        }

        return result_df, summary
