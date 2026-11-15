"""
ColdGuard AI - AWS S3 Integration Module
Manages Amazon S3 operations for dataset storage, model artifacts, and inference outputs.
Designed with beginner-friendly validation and transparent status reporting.
"""

import io
import os
from typing import Tuple, Dict, Any, Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError


class S3Manager:
    """
    Manages Amazon S3 interactions for ColdGuard AI.
    Never hardcodes secrets. Uses environment variables or standard AWS credential chains.
    """

    def __init__(
        self,
        bucket_name: Optional[str] = None,
        region_name: str = "ap-south-1"
    ):
        self.bucket_name = bucket_name or os.getenv("COLDGUARD_S3_BUCKET", "")
        self.region_name = region_name or os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
        self._s3_client = None

    def get_client(self):
        """Initializes and returns boto3 S3 client."""
        if self._s3_client is None:
            self._s3_client = boto3.client("s3", region_name=self.region_name)
        return self._s3_client

    def check_connection(self) -> Dict[str, Any]:
        """
        Tests whether valid AWS credentials exist and whether the configured S3 bucket is accessible.
        Never fakes a successful connection.
        """
        status: Dict[str, Any] = {
            "has_credentials": False,
            "bucket_accessible": False,
            "caller_arn": None,
            "bucket_name": self.bucket_name,
            "region": self.region_name,
            "error_message": None,
            "instructions": []
        }

        # Step 1: Check STS identity to verify AWS credentials
        try:
            sts_client = boto3.client("sts", region_name=self.region_name)
            identity = sts_client.get_caller_identity()
            status["has_credentials"] = True
            status["caller_arn"] = identity.get("Arn")
        except (NoCredentialsError, PartialCredentialsError):
            status["error_message"] = "No AWS credentials found."
            status["instructions"] = [
                "Configure AWS credentials via AWS CLI with 'aws configure' or set AWS_ACCESS_KEY_ID in .env.",
                "Ensure your IAM user or role has permissions for Amazon S3 and Amazon SageMaker AI."
            ]
            return status
        except Exception as e:
            status["error_message"] = f"Failed to authenticate with AWS: {str(e)}"
            status["instructions"] = [
                "Verify your network connection and AWS region configuration.",
                f"Configured region: {self.region_name}"
            ]
            return status

        # Step 2: Check S3 bucket if configured
        if not self.bucket_name:
            status["error_message"] = "S3 Bucket name is not configured."
            status["instructions"] = [
                "Provide a globally unique bucket name in .env (COLDGUARD_S3_BUCKET=<your-bucket>).",
                "Ensure the bucket exists in the specified AWS region."
            ]
            return status

        try:
            s3 = self.get_client()
            s3.head_bucket(Bucket=self.bucket_name)
            status["bucket_accessible"] = True
        except ClientError as e:
            error_code = e.response.get("Error", {}).get("Code", "")
            if error_code == "404":
                status["error_message"] = f"Bucket '{self.bucket_name}' does not exist."
                status["instructions"] = [
                    f"Create bucket '{self.bucket_name}' in region '{self.region_name}'.",
                    "Or use an existing bucket with proper S3 permissions."
                ]
            elif error_code == "403":
                status["error_message"] = f"Access denied to bucket '{self.bucket_name}' (HTTP 403)."
                status["instructions"] = [
                    "Verify your IAM policy grants 's3:ListBucket', 's3:GetObject', and 's3:PutObject'.",
                    f"Bucket ARN: arn:aws:s3:::{self.bucket_name}"
                ]
            else:
                status["error_message"] = f"S3 error ({error_code}): {str(e)}"
        except Exception as e:
            status["error_message"] = f"Unexpected S3 error: {str(e)}"

        return status

    def upload_dataframe_as_csv(
        self,
        df,
        s3_key: str,
        include_header: bool = True
    ) -> Tuple[bool, str]:
        """
        Uploads a Pandas DataFrame directly to S3 as a CSV without local disk writing.
        """
        try:
            s3 = self.get_client()
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, header=include_header)
            
            s3.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=csv_buffer.getvalue().encode("utf-8")
            )
            s3_uri = f"s3://{self.bucket_name}/{s3_key}"
            return True, s3_uri
        except Exception as e:
            return False, f"Failed to upload to S3 ({s3_key}): {str(e)}"

    def download_s3_file_as_string(self, s3_key: str) -> Tuple[bool, str]:
        """
        Downloads a text/CSV file from S3 and returns content as string.
        """
        try:
            s3 = self.get_client()
            obj = s3.get_object(Bucket=self.bucket_name, Key=s3_key)
            content = obj["Body"].read().decode("utf-8")
            return True, content
        except Exception as e:
            return False, f"Failed to download S3 key ({s3_key}): {str(e)}"
