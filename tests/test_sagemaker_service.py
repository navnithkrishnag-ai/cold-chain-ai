"""
Unit tests for SageMaker RCF service, batch transform config, JSON Lines parsing,
cloud prerequisite validation, and feature scaling without live AWS credentials.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from src.sagemaker_service import (
    SageMakerRCFService,
    parse_rcf_jsonlines_scores,
    split_chronological_train_eval,
    RCF_IMAGE_URIS
)


def test_rcf_training_channel_sharded_by_s3_key():
    """Verify RCF training channel explicitly configures ShardedByS3Key."""
    service = SageMakerRCFService(
        bucket_name="test-bucket",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        region_name="ap-south-1"
    )

    params = service.build_training_job_params(
        job_name="test-job",
        train_s3_uri="s3://test-bucket/train.csv",
        output_s3_uri="s3://test-bucket/output/"
    )

    channels = params["InputDataConfig"]
    assert len(channels) == 1
    assert channels[0]["ChannelName"] == "train"
    s3_data_source = channels[0]["DataSource"]["S3DataSource"]
    assert s3_data_source["S3DataDistributionType"] == "ShardedByS3Key"


def test_batch_transform_accept_jsonlines():
    """Verify TransformOutput explicitly sets Accept = application/jsonlines."""
    service = SageMakerRCFService(
        bucket_name="test-bucket",
        role_arn="arn:aws:iam::123456789012:role/TestRole",
        region_name="ap-south-1"
    )

    transform_params = service.build_transform_job_params(
        transform_job_name="test-transform",
        model_name="test-model",
        eval_s3_uri="s3://test-bucket/eval.csv",
        transform_output_s3="s3://test-bucket/output/"
    )

    transform_output = transform_params["TransformOutput"]
    assert transform_output["Accept"] == "application/jsonlines"
    assert transform_output["AssembleWith"] == "Line"


def test_jsonlines_score_parsing_valid():
    """Verify correct parsing of valid JSON Lines score objects."""
    raw_output = '{"score": 0.45}\n{"score": 1.28}\n{"score": 12.95}\n'
    scores = parse_rcf_jsonlines_scores(raw_output, expected_count=3)
    assert len(scores) == 3
    assert scores == [pytest.approx(0.45), pytest.approx(1.28), pytest.approx(12.95)]


def test_jsonlines_score_parsing_mismatch_raises_error():
    """Verify score count mismatch raises ValueError."""
    raw_output = '{"score": 0.45}\n{"score": 1.28}\n'
    with pytest.raises(ValueError, match="Score count mismatch"):
        parse_rcf_jsonlines_scores(raw_output, expected_count=3)


def test_jsonlines_score_parsing_missing_score_raises_error():
    """Verify missing score key is NOT silently replaced with zero."""
    raw_output = '{"score": 0.45}\n{"other_key": 1.28}\n'
    with pytest.raises(ValueError, match="missing 'score' key"):
        parse_rcf_jsonlines_scores(raw_output, expected_count=2)


def test_jsonlines_score_parsing_invalid_score_raises_error():
    """Verify non-numeric score raises clear error."""
    raw_output = '{"score": 0.45}\n{"score": "invalid_value"}\n'
    with pytest.raises(ValueError, match="contains non-numeric score"):
        parse_rcf_jsonlines_scores(raw_output, expected_count=2)


def test_s3_prerequisite_validation_all_pass():
    """Verify validate_cloud_prerequisites passes when STS, S3, Role, and SageMaker succeed."""
    service = SageMakerRCFService(
        bucket_name="my-coldguard-bucket",
        role_arn="arn:aws:iam::123456789012:role/ColdGuardRole",
        region_name="ap-south-1"
    )

    mock_s3_mgr = MagicMock()
    mock_s3_mgr.check_connection.return_value = {
        "has_credentials": True,
        "bucket_accessible": True,
        "caller_arn": "arn:aws:iam::123456789012:user/tester"
    }

    mock_sm_client = MagicMock()
    mock_sm_client.list_training_jobs.return_value = {"TrainingJobSummaries": []}

    with patch.object(service, "get_client", return_value=mock_sm_client):
        prereqs = service.validate_cloud_prerequisites(s3_manager=mock_s3_mgr)

    assert prereqs["ready"] is True
    assert len(prereqs["issues"]) == 0


def test_s3_prerequisite_validation_fails_on_inaccessible_bucket():
    """Verify validation fails if S3 bucket does not exist or is inaccessible."""
    service = SageMakerRCFService(
        bucket_name="non-existent-bucket",
        role_arn="arn:aws:iam::123456789012:role/ColdGuardRole",
        region_name="ap-south-1"
    )

    mock_s3_mgr = MagicMock()
    mock_s3_mgr.check_connection.return_value = {
        "has_credentials": True,
        "bucket_accessible": False,
        "error_message": "Bucket 'non-existent-bucket' does not exist."
    }

    prereqs = service.validate_cloud_prerequisites(s3_manager=mock_s3_mgr)
    assert prereqs["ready"] is False
    assert any("does not exist" in issue for issue in prereqs["issues"])


def test_s3_prerequisite_validation_fails_on_missing_role():
    """Verify validation fails if IAM execution role ARN is omitted."""
    service = SageMakerRCFService(
        bucket_name="my-bucket",
        role_arn="",  # Missing role ARN
        region_name="ap-south-1"
    )

    mock_s3_mgr = MagicMock()
    mock_s3_mgr.check_connection.return_value = {
        "has_credentials": True,
        "bucket_accessible": True
    }

    prereqs = service.validate_cloud_prerequisites(s3_manager=mock_s3_mgr)
    assert prereqs["ready"] is False
    assert any("COLDGUARD_SAGEMAKER_ROLE_ARN" in issue for issue in prereqs["issues"])


def test_consistent_feature_scaling():
    """Verify StandardScaler is fitted on train only and consistently scales eval data."""
    service = SageMakerRCFService(region_name="ap-south-1")

    # Construct train dataset with high humidity range and low temperature change
    train_df = pd.DataFrame({
        "temperature": [4.0, 5.0, 6.0, 7.0],
        "humidity": [20.0, 40.0, 60.0, 80.0],
        "temperature_change": [0.1, -0.1, 0.2, 0.0],
        "temperature_deviation": [0.5, -0.5, 1.0, -1.0]
    })

    # Eval dataset
    eval_df = pd.DataFrame({
        "temperature": [5.5, 6.5],
        "humidity": [50.0, 70.0],
        "temperature_change": [0.0, 0.1],
        "temperature_deviation": [0.0, 0.2]
    })

    # Fit on training data
    train_scaled = service.prepare_scaled_features(train_df, is_training=True)
    assert service.scaler is not None
    assert train_scaled.shape == (4, 4)
    # Means of train scaled should be close to 0
    assert np.allclose(train_scaled.mean(), 0.0, atol=1e-5)

    # Transform eval data using already fitted scaler
    eval_scaled = service.prepare_scaled_features(eval_df, is_training=False)
    assert eval_scaled.shape == (2, 4)
    # Check that eval_scaled used the exact training mean and scale
    expected_temp_scaled = (eval_df["temperature"] - train_df["temperature"].mean()) / train_df["temperature"].std(ddof=0)
    assert np.allclose(eval_scaled["temperature"], expected_temp_scaled)


def test_chronological_train_eval_split():
    """Verify chronological train/eval split preserves order, timestamps, and sensor IDs."""
    ts = pd.date_range("2024-09-02 08:00:00", periods=10, freq="1min")
    df = pd.DataFrame({
        "timestamp": ts,
        "sensor_id": ["S1", "S2"] * 5,
        "temperature": np.linspace(4.0, 10.0, 10),
        "humidity": [75.0] * 10,
        "temperature_change": [0.0] * 10,
        "temperature_deviation": [0.0] * 10
    })

    df_train, df_eval = split_chronological_train_eval(df, train_ratio=0.7)
    assert len(df_train) == 7
    assert len(df_eval) == 3
    # Check chronological boundary
    assert df_train["timestamp"].max() < df_eval["timestamp"].min()
    assert "sensor_id" in df_train.columns
    assert "sensor_id" in df_eval.columns


def test_unsupported_region_raises_error():
    """Verify unsupported AWS regions raise ValueError instead of constructing invalid ECR URIs."""
    service = SageMakerRCFService(region_name="mars-east-1")
    with pytest.raises(ValueError, match="Unsupported AWS region"):
        service.get_rcf_image_uri()
