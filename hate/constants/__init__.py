import os

from datetime import datetime

#Common constants
TIMESTAMP : str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME = "hatespeech25"
ZIP_FILE_NAME = 'dataset.zip'
LABEL = 'label'
TWEET = 'tweet'

#Data Ingestion related constants
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATA_INGESTION_IMBALANCE_DATA_DIR = "imbalanced_data.csv"
DATA_INGESTION_RAW_DATA_DIR = "raw_data.csv"

#Data Validation related constants



#Data Transformation related constants
DATA_TRANSFORMATION_ARTIFACTS_DIR = 'DataTansformationArtifacts'
TRANSFORMED_FILE_NAME = 'final.csv'
DATA_DIR = 'data'
ID = 'id'
AXIS = 1
INPLACE = True
DROP_COLUMNS = ['Unnamed: 0', 'count', 'hate_speech', 'offensive_language', 'neither']
CLASS = 'class'