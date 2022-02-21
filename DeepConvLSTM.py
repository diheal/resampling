from tensorflow.keras.layers import *
from tensorflow.keras.models import *

def get_DCL(n_timesteps, n_features):
    input1 = Input((n_timesteps, n_features))
    h1 = Conv1D(filters=128, kernel_size=5, activation='relu', name='conv1_1')(input1)
    h2 = Conv1D(filters=128, kernel_size=5, activation='relu', name='conv1_2')(h1)
    h3 = Conv1D(filters=128, kernel_size=5, activation='relu', name='conv1_3')(h2)
    h4 = Conv1D(filters=128, kernel_size=5, activation='relu', name='conv1_4')(h3)
    h = LSTM(128, return_sequences=True)(h4)
    h = LSTM(128, return_sequences=False)(h)
    return Model(input1, h)