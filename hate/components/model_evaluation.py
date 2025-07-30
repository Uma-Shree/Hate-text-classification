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

            self.gcloud.sync_folder_from_gcloud(self.model_evaluation_config.BUCKET_NAME,
                                                self.model_evaluation_config.MODEL_NAME,
                                                self.model_evaluation_config.BEST_MODEL_DIR_PATH)
            
            best_model_path = os.path.join(self.model_evaluation_config.BEST_MODEL_DIR_PATH,
                                           self.model_evaluaition_config.MODEL_NAME)
            
            logging.info("Exited best_model_from_gcloud method of model evaluation component")

            return best_model_path

        except Exception as e:
            raise CustomException(e, sys) from e

    def evaluate(self):
        '''
        :param model :currently trained model or best model from gcloud storage
        :param data_loader : Data loader for validation dataset
        return : loss
        
        '''
        try:
            logging.info("Entered evaluate method of model evaluation component")

            print(self.model__trainer_artifacts.x_test_path)

            x_test = pd.read_csv(self.model_trainer_artifacts.x_test_path, index_col = 0)
            print(x_test)
            y_test = pd.read_csv(self.model_trainer_artifacts.y_test_path, index_col = 0)

            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            
            load_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)

            x_test = x_test['tweet'].astype(str)

            x_test = x_test.squeeze()
            y_test = y_test.squeeze()

            test_sequences = tokenizer.texts_to_sequences(x_test)
            test_sequences_matrix = pad_sequences(test_sequences, maxlen = MAX_LEN)

            print(f"-----------------------{test_sequences_matrix.shape}-----------------------")

            print(f"-----------------------{x_test.shape}-----------------------")
            print(f"-----------------------{y_test.shape}-----------------------")

            accuracy = load_model.evaluate(test_sequences_matrix, y_test)
            logging.info(f"the test accuracy is : {accuracy}" )

            lstm_prediction = load_model.predict(test_sequences_matrix)
            res = []
            for prediction in lstm_prediction:
                if prediction > 0.5:
                    res.append(1)
                else:
                    res.append(0)

            print(confusion_matrix(y_test, res))

            logging.info("Exited evaluate method of model evaluation component")
            return accuracy

        except Exception as e:
            raise CustomException(e, sys) from e
        

    def initiate_model_evaluation(self) -> ModelEvaluationArtifacts:
        """
        Method Name: initiate_model_evaluation
        Description: This method initiates the model evaluation steps
        Output : Returns model evaluation artifacts
        On Failure: Raise exception
        """
        try:
            logging.info("Entered initiate_model_evaluation method of model evaluation component")

            trained_model = keras.models.load_model(self.model_trainer_artifacts.trained_model_path)
            with open('tokenizer.pickle', 'rb') as handle:
                load_tokenizer = pickle.load(handle)

            trained_model_accuracy = self.evaluate()

            logging.info("fetch best model from gcloud")

            best_model_path = self.get_best_model_from_gcloud()

            logging.info("check if the best model present in the gcloud or not?")

            if os.path.isfile(best_model_path) is False:
                is_model_accepted = Truel

                logging.info("gcloud storage model is false and currently trained model accepted is true")

            else:
                logging.info("load the best fetched from gcloud storage")
                best_model_accuracy = self.evaluate()

                logging.info("comparing the loss betwween best_model_loss and trained_model_loss")

                if best_model_accuracy > trained_model_accuracy:
                    is_model_accepted = True
                    logging.info("trained model not accepted")
                else:
                    is_model_accepted = False
                    logging.info("trained model accepted")
            
            model_evaulation_artifacts = ModelEvaluationArtifacts(
                is_model_accepted=is_model_accepted
                
            )
            logging.info(f"model evaluation artifacts {model_evaulation_artifacts}")
            logging.info("returning model evaluation artifacts")
            logging.info("Exited initiate_model_evaluation method of model evaluation component")

            return model_evaulation_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e

        