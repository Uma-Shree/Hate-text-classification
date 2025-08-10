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
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt

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
        
            #DATA VALIDATION
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
        
            #DATA CLEANING
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

    def hyperparameter_tuning(self, sequences_matrix, y_train):
        """Perform hyperparameter tuning"""
        try:
            logging.info("Starting hyperparameter tuning")
            
            model_architecture = ModelArchitecture()
            
            # Create tuner
            tuner = kt.RandomSearch(
                model_architecture.build_model,
                objective='val_accuracy',
                max_trials=self.model_trainer_config.MAX_TRIALS,
                executions_per_trial=self.model_trainer_config.EXECUTIONS_PER_TRIAL,
                directory=self.model_trainer_config.TUNER_DIRECTORY,
                project_name=self.model_trainer_config.TUNER_PROJECT_NAME
            )
            
            # Early stopping for tuning
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=3,
                restore_best_weights=True,
                verbose=1
            )
            
            # Search for best hyperparameters
            tuner.search(
                sequences_matrix, y_train,
                epochs=self.model_trainer_config.EPOCH,
                validation_split=self.model_trainer_config.VALIDATION_SPLIT,
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Get best model
            best_model = tuner.get_best_models(num_models=1)[0]
            best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
            
            logging.info(f"Best hyperparameters: {best_hyperparameters.values}")
            
            return best_model, best_hyperparameters
            
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        try:
            logging.info("Entered the initiate_model_trainer method of ModelTrainer class")

            X_train, X_test, y_train, y_test = self.spliting_data(
                csv_path=self.data_transformation_artifacts.transformed_data_path
            )
            
            sequences_matrix, tokenizer = self.tokenizing(X_train)
            
            if self.model_trainer_config.HYPERPARAMETER_TUNING:
                # Use hyperparameter tuning
                model, best_hyperparameters = self.hyperparameter_tuning(sequences_matrix, y_train)
                
                # Save hyperparameters
                hyperparams_path = os.path.join(self.model_trainer_config.TRAINED_MODEL_DIR, 'best_hyperparameters.json')
                with open(hyperparams_path, 'w') as f:
                    import json
                    json.dump(best_hyperparameters.values, f, indent=2)
                    
            else:
                # Use default model
                model_architecture = ModelArchitecture()
                model = model_architecture.get_model()
                
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True,
                    verbose=1
                )
                
                model.fit(
                    sequences_matrix, y_train,
                    batch_size=self.model_trainer_config.BATCH_SIZE,
                    epochs=self.model_trainer_config.EPOCH,
                    validation_split=self.model_trainer_config.VALIDATION_SPLIT,
                    callbacks=[early_stopping]
                )
            
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


   
        


            

