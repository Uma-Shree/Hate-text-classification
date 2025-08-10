#creating model architecture (same as in notebook)

import keras_tuner as kt
from keras.models import Sequential
from keras.optimizers import RMSprop, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout, Embedding, SpatialDropout1D
from hate.constants import *

class ModelArchitecture:

    def __init__(self):
        pass

    def build_model(self, hp):
        """Build model with hyperparameters for tuning"""
        model = Sequential()
        
        # Tune embedding dimension
        embedding_dim = hp.Int('embedding_dim', min_value=50, max_value=200, step=50)
        model.add(Embedding(MAX_WORDS, embedding_dim, input_length=MAX_LEN))
        
        # Tune spatial dropout
        spatial_dropout = hp.Float('spatial_dropout', min_value=0.1, max_value=0.5, step=0.1)
        model.add(SpatialDropout1D(spatial_dropout))
        
        # Tune LSTM units
        lstm_units = hp.Int('lstm_units', min_value=64, max_value=256, step=32)
        
        # Tune LSTM dropout
        lstm_dropout = hp.Float('lstm_dropout', min_value=0.1, max_value=0.5, step=0.1)
        recurrent_dropout = hp.Float('recurrent_dropout', min_value=0.1, max_value=0.5, step=0.1)
        
        model.add(LSTM(lstm_units, dropout=lstm_dropout, recurrent_dropout=recurrent_dropout))
        
        # Optional: Add dense layer
        if hp.Boolean('add_dense'):
            dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=32)
            dense_dropout = hp.Float('dense_dropout', min_value=0.1, max_value=0.5, step=0.1)
            model.add(Dense(dense_units, activation='relu'))
            model.add(Dropout(dense_dropout))
        
        model.add(Dense(1, activation=ACTIVATION))
        
        # Tune optimizer
        optimizer_choice = hp.Choice('optimizer', ['rmsprop', 'adam'])
        learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')
        
        if optimizer_choice == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        else:
            optimizer = Adam(learning_rate=learning_rate)
        
        model.compile(loss=LOSS, optimizer=optimizer, metrics=METRICS)
        return model

    def get_model(self):
        """Original method for backward compatibility"""
        model = Sequential()
        model.add(Embedding(MAX_WORDS, 100, input_length=MAX_LEN))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(1, activation=ACTIVATION))
        model.compile(loss=LOSS, optimizer=RMSprop(), metrics=METRICS)
        return model
