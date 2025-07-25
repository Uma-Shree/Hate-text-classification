import sys

from hate.logger import logging
from hate.exception import CustomException
from hate.components.data_ingestion import DataIngestion
from hate.components.data_transformation import DataTransformation
from hate.entity.config_entity import (DataIngestionConfig, DataTransformationConfig)
from hate.entity.artifact_entity import (DataIngestionArtifacts, DataTransformationArtifacts)

class TrainPipeline:
    def __init__(self):
        
        try:
            logging.info(f"{'='*20}{'TRAINING PIPELINE LOGGING STARTED'}{'='*20}")
            self.data_ingestion_config = DataIngestionConfig()
            self.data_transformation_config = DataTransformationConfig()

        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifacts:
        logging.info("Entered the start_data_ingestion method of TrainPipeline class")
        try:
            logging.info("Getting the data from GCloud storage bucket")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info("Got the data from GCloud storage bucket")
            logging.info("Exited the start_data_ingestion method of TrainPipeline class")
            return data_ingestion_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e

    def start_data_transformation(self, data_ingestion_artifacts = DataIngestionArtifacts) -> DataTransformationArtifacts:
        logging.info("")
        try:
            data_transformation = DataTransformation(
                data_ingestion_artifacts = data_ingestion_artifacts,
                data_transformation_config = self.data_transformation_config
            )

            data_transformation_artifacts = data_transformation.initiate_data_transformation()

            logging.info("Exited the start_data_transformation method of TrainPipeline class")
            return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e
        
    
    def run_pipeline(self):
        logging.info("Entered the run_pipeline method of TrainPipeline class")
        try:
            data_ingestion_artifacts = self.start_data_ingestion()

            data_transformation_artifacts = self.start_data_transformation(
                data_ingestion_artifacts = data_ingestion_artifacts
            )
            logging.info("Exited the run_pipeline method of TrainPipeline class")

        except Exception as e:
            raise CustomException(e, sys) from e
        

