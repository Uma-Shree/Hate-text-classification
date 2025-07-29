import os
import sys
import keras
import pickle
import numpy as np
import pandas as pd
from keras.utils import pad_sequences
from hate.logger import logging
from hate.exception import CustomException
from hate.constants import *
from hate.configuation.gcloud_syncer import GCloudSync
from sklearn.metrics import confusion_matrix
from hate.entity.config_entity import ModelEvaluationConfig
from hate.entity.artifact_entity import ModelEvaluationArtifacts, ModelTrainerArtifacts, DataTransformationArtifacts


class ModelEvaluation:

    def __init__(self, model_evaluation_config: ModelEvaluationConfig,
                 model_trainer_artifacts: ModelTrainerArtifacts,
                 data_transformation_artifacts: DataTransformationArtifacts):
        """
        : param model_evaluation_config: Configuration for model evaluation
        : param model_trainer_artifacts: Artifacts from model trainer
        : param data_transformation_artifacts: Artifacts from data transformation
        """
        self.model_evaluation_config = model_evaluation_config
        self.model_trainer_artifacts = model_trainer_artifacts
        self.data_transformation_artifacts = data_transformation_artifacts
        self.gcloud = GCloudSync()

    
    def best_model_from_gcloud(self) -> str:
        """
            fetch best model from gcloud storage and store inside best model directory path

        """

        try:
            logging.info("Entered best_model_from_gcloud method of model evaluation component")

            os.makedirs(self.model_evaluation_config.BEST_MODEL_DIR_PATH, exist_ok=True)

            self.gcloud.syc

            
        
