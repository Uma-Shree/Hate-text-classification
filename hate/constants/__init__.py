import os

from datetime import datetime

#Common constants
TIMESTAMP : str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACT_DIR = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME = "hatespeech25"
ZIP_FILE_NAME = 'dataset.zip'
LABEL = 'label'
TWEET = 'tweet'

#Data Ingestion related constants
DATA_INGESTION_ARTIFACTS_DIR = "dataIngestionArtifacts"
DATA_INGESTION_IMBALANCE_DATA_DIR = "imbalanced_data.csv"
DATA_INGESTION_RAW_DATA_DIR = "raw_data.csv"