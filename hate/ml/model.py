#creating model architecture (same as in notebook)

from hate.entity.config_entity import ModelTrainerConfig
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding, SpatialDropout1D
from hate.constants import *

class ModelArchitecture:

    def __init__(self):
        pass

    def get_model(self):
        model = Sequential()
        model.add(Embedding(MAX_WORDS, 100, input_length = MAX_LEN)) #100 dimension
        model.add(SpatialDropout1D(0.2)) #earlier 0.2
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2)) #earlier 0.2 both
        model.add(Dense(1, activation= ACTIVATION )) #at last layer
        model.summary()
        model.compile(loss=LOSS, optimizer = RMSprop(), metrics=METRICS)

        return model