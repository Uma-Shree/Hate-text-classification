import os 
import sys
import pickle
import pandas as pd
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences

from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts
from hate.ml.model import ModelArchitecture

class ModelTrainer:
    def __init__(self, data_transformation_artifacts: DataTransformationArtifacts,
                 model_trainer_config: ModelTrainerConfig):
        
            self.data_transformation_artifacts = data_transformation_artifacts
            self.model_trainer_config = model_trainer_config
           
        
    def spliting_data(self, csv_path):
        try:
            logging.info("Splitting the data into train and test")
            logging.info("Reading the csv file data")
            df = pd.read_csv(csv_path, index_col = False)
            logging.info("Splitting the data into train and test")
            X = df[TWEET]
            y = df[LABEL]

            logging.info("Applying train test split")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
            print(len(X_train), len(X_test), len(y_train), len(y_test))
            print(type(X_train), type(X_test), type(y_train), type(y_test))
            logging.info("Exiting the spliting_data method of ModelTrainer class")

            return X_train, X_test, y_train, y_test

        except Exception as e:
            raise CustomException(e, sys) from e
          

    
          
    '''
    def tokenizing(self,X_train):
        try:
            logging.info("Tokenizing the data")
            tokenizer = Tokenizer(num_words = self.model_trainer_config.MAX_WORDS)
            print(f"NaN count: {X_train.isna().sum()}")
            tokenizer.fit_on_texts(X_train)
            sequences = tokenizer.texts_to_sequences(X_train)
            sequences_matrix = pad_sequences(sequences, maxlen = self.model_trainer_config.MAX_LEN)
            logging.info("The sequence matrix is : {sequences_matrix}")

            return sequences_matrix, tokenizer

        except Exception as e:
            raise CustomException(e, sys) from e
        
        '''
    
    def tokenizing(self, X_train):
        try:
            logging.info("Tokenizing the data")
        
            # === DATA VALIDATION ===
            print("=== DATA VALIDATION ===")
            print(f"Original data shape: {X_train.shape}")
            print(f"Data type: {type(X_train)}")
        
            # Check for NaN values
            nan_count = X_train.isna().sum()
            print(f"NaN count: {nan_count}")
        
            # Check data types of sample values
            sample_types = [type(x) for x in X_train.head(10) if pd.notna(x)]
            print(f"Sample value types: {sample_types}")
        
            # Show sample values
            print(f"Sample values: {list(X_train.head())}")
        
            # Count non-string values
            non_string_count = sum(1 for x in X_train if not isinstance(x, str) and pd.notna(x))
            print(f"Non-string values count: {non_string_count}")
        
            # === DATA CLEANING ===
            logging.info("Cleaning data before tokenization")
        
            # Convert to pandas Series if not already
            if not isinstance(X_train, pd.Series):
                X_train = pd.Series(X_train)
        
            # Remove NaN values
            original_length = len(X_train)
            X_train = X_train.dropna()
            removed_nan = original_length - len(X_train)
            print(f"Removed {removed_nan} NaN values")
        
            # Convert all values to strings
            X_train = X_train.astype(str)
        
            # Remove empty strings and whitespace-only strings
            X_train = X_train[X_train.str.strip() != '']
        
            # Final validation
            print(f"Final cleaned data shape: {X_train.shape}")
            print(f"Final sample values: {list(X_train.head())}")
            print("=== VALIDATION COMPLETE ===\n")
        
            # Proceed with tokenization
            logging.info("Starting tokenization process")
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(X_train)
        
            sequences = tokenizer.texts_to_sequences(X_train)
            sequences_matrix = pad_sequences(sequences, maxlen=self.model_trainer_config.MAX_LEN)
        
            logging.info(f"Tokenization completed. Sequences matrix shape: {sequences_matrix.shape}")
            logging.info(f"The sequence matrix is: {sequences_matrix}")

            return sequences_matrix, tokenizer

        except Exception as e:
            raise CustomException(e, sys) from e


    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        logging.info("Entered the initiate_model_trainer method of ModelTrainer class")

        """
        Method Name: initiate_model_trainer
        Description: This method initiates the model trainer steps
        
        Output : Returns model trainer artifacts
        On Failure: Raise exception
        """

        try:
            logging.info("Entered the initiate_model_trainer method of ModelTrainer class")

            X_train, X_test, y_train, y_test = self.spliting_data(csv_path= self.data_transformation_artifacts.transformed_data_path)
            model_architecture = ModelArchitecture()
            model = model_architecture.get_model()

            logging.info(f"X_train size is: {X_train.shape}")
            logging.info(f"y_train size is: {y_train.shape}")
            

            sequences_matrix, tokenizer = self.tokenizing(X_train)

            logging.info("Entering the model training")
            model.fit(sequences_matrix,
                      y_train,
                      batch_size=self.model_trainer_config.BATCH_SIZE, 
                      epochs = self.model_trainer_config.EPOCH,
                      validation_split = self.model_trainer_config.VALIDATION_SPLIT,
                      )
            logging.info("Exited the model training")

            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            

            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)

            logging.info("Saving the model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            X_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)
            logging.info("Model saved successfully")

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path=self.model_trainer_config.TRAINED_MODEL_PATH,
                X_test_path=self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path=self.model_trainer_config.Y_TEST_DATA_PATH
            )

            logging.info(f"Model trainer artifacts: {model_trainer_artifacts}")
            
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e


   
        


            

