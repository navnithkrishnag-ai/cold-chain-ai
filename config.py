"""
ColdGuard AI - Configuration Module
Manages application settings, AWS configuration, cold-chain presets, and execution modes.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_DIR = DATA_DIR / "sample"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Ensure directories exist
SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load .env if present
load_dotenv(dotenv_path=BASE_DIR / ".env")

# App Brand Info
APP_NAME = "ColdGuard AI"
APP_TAGLINE = "AI-Enabled Cold-Chain Condition Monitoring & Early Warning System"
APP_ICON = "❄️"

# AWS Configuration
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
S3_BUCKET = os.getenv("COLDGUARD_S3_BUCKET", "")
SAGEMAKER_ROLE_ARN = os.getenv("COLDGUARD_SAGEMAKER_ROLE_ARN", "")
EXECUTION_MODE = os.getenv("COLDGUARD_EXECUTION_MODE", "LOCAL").upper()

# Default Cold-Chain Preset (Configurable Chilled Goods / Vaccines / Dairy)
# Note: 2°C to 8°C is an illustrative operating range for demonstration, not universal for all cold goods.
DEFAULT_MIN_TEMP = float(os.getenv("DEFAULT_MIN_TEMP", "2.0"))
DEFAULT_MAX_TEMP = float(os.getenv("DEFAULT_MAX_TEMP", "8.0"))
DEFAULT_MAX_EXCURSION_MINUTES = float(os.getenv("DEFAULT_MAX_EXCURSION_MINUTES", "30.0"))
DEFAULT_ANOMALY_SIGMA = float(os.getenv("DEFAULT_ANOMALY_SIGMA", "3.0"))

# S3 Prefix Names
S3_PREFIX_RAW = "coldguard/raw_data/"
S3_PREFIX_PROCESSED = "coldguard/processed_data/"
S3_PREFIX_MODELS = "coldguard/models/"
S3_PREFIX_INFERENCE = "coldguard/inference_output/"

# SageMaker RCF Settings
RCF_CONTAINER_REGISTRY_AP_SOUTH_1 = "991648021394.dkr.ecr.ap-south-1.amazonaws.com/randomcutforest:latest"

# Color Palette (Dark navy / steel blue theme)
THEME_COLORS = {
    "primary": "#1E3E62",
    "background_dark": "#0B192C",
    "card_bg": "#12253B",
    "text": "#E2E8F0",
    "text_muted": "#94A3B8",
    "accent_blue": "#38BDF8",
    "status_low": "#10B981",       # Emerald Green
    "status_moderate": "#F59E0B",  # Amber
    "status_high": "#F97316",      # Orange
    "status_critical": "#EF4444",  # Crimson Red
}
