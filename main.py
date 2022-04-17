import opensmile
import pandas as pd
import csv
import os
import numpy as np
# from torch import int64
import librosa
import soundfile as sf

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
MODE = "TEST"

######################
# Feature Extraction #
######################
def trim_audio_data(audio_file):
    sr = 96000
    sec = 300

    y, sr = librosa.load(audio_file, sr=sr)

    ny = y[:sr*sec]

    sf.write(file=audio_file, data=ny, samplerate=sr)

audio_file = "audios/class.wav"
trim_audio_data(audio_file)
duration = librosa.get_duration(filename=audio_file)
num_batch = int(duration / 5)
print("audio_file:", audio_file)
print("duration: {:.2f}s".format(duration))
print("num_batch:", num_batch)
file_name = "features/" + audio_file.split("/")[-1].split(".")[0] + ".npy"


if os.path.isfile(file_name):
    print(file_name + " already exists!")
    print("Loading the file ...")

    LLDs = np.load(file_name)

else:
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.ComParE_2016, #emobase, #eGeMAPSv02, #eGeMAPSv01b, #eGeMAPSv01a, #ComParE_2016, #GeMAPSv01b,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,  #Functionals,
    )
    LLDs = smile.process_file(audio_file)
    LLD_names = smile.feature_names
    LLDs = LLDs.to_numpy()
    np.save(file_name, LLDs)

LLDs = np.append(LLDs, LLDs[-4:], axis=0)

print("LLDs shape:", LLDs.shape)
L = int(LLDs.shape[0] / num_batch)
print("L:", L)
LLDs = LLDs.reshape(num_batch, 65, L)

input_shape = LLDs.shape

#########
# CRDNN #
#########
import pandas as pd
import matplotlib.pyplot as plt
import keras

from keras.layers import Dense, Input, GlobalMaxPooling1D, Reshape, LSTM, Activation, GRU
from keras.layers import Conv1D, MaxPooling1D, Embedding, Concatenate, AveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten
from keras.layers import GlobalMaxPooling2D, Conv3D, MaxPooling3D, ConvLSTM2D, Add
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf

class CRDNN(keras.Model):
    def __init__(self,):
        super().__init__()
        # CNN
        self.conv1 = Conv1D(L, kernel_size=8)
        self.maxpooling1 = MaxPooling1D(3)
        self.activation1 = Activation('relu')
        self.bn1 = BatchNormalization()
        # RNN
        self.gru1 = GRU(128, return_sequences=True)
        self.gru2 = GRU(L,return_sequences=True)
        # Pooling
        self.mean1 = AveragePooling1D()
        self.flatten = Flatten()
        # FC
        self.dense1 = Dense(2)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpooling1(x)
        x = self.activation1(x)
        f = self.bn1(x)
        h = self.gru1(f)
        h = self.gru2(h)
        z = self.mean1(h)
        z = self.flatten(z)
        output = self.dense1(z)

        return output

model = CRDNN()
model.build(input_shape)

# Model Compile
def ccc(y_true, y_pred):
    '''Lin's Concordance correlation coefficient: https://en.wikipedia.org/wiki/Concordance_correlation_coefficient
    
    The concordance correlation coefficient is the correlation between two variables that fall on the 45 degree line through the origin.
    
    It is a product of
    - precision (Pearson correlation coefficient) and
    - accuracy (closeness to 45 degree line)

    Interpretation:
    - `rho_c =  1` : perfect agreement
    - `rho_c =  0` : no agreement
    - `rho_c = -1` : perfect disagreement 
    
    Args: 
    - y_true: ground truth
    - y_pred: predicted values
    
    Returns:
    - concordance correlation coefficient (float)
    '''
    
    import keras.backend as K 
    # covariance between y_true and y_pred
    N = K.int_shape(y_pred)[-1]
    s_xy = 1.0 / (N - 1.0 + K.epsilon()) * K.sum((y_true - K.mean(y_true)) * (y_pred - K.mean(y_pred)))
    # means
    x_m = K.mean(y_true)
    y_m = K.mean(y_pred)
    # variances
    s_x_sq = K.var(y_true)
    s_y_sq = K.var(y_pred)
    
    # condordance correlation coefficient
    ccc = (2.0*s_xy) / (s_x_sq + s_y_sq + (x_m-y_m)**2)
    
    return ccc


model.compile(optimizer=Adam(learning_rate=1e-4), loss=ccc)
early_stop = EarlyStopping(monitor = 'loss', min_delta = 0.001, patience = 10, mode = 'auto', verbose = 1)

#########
# Train #
#########
import random

model_path = "model/" + audio_file.split("/")[-1].split(".")[0] + "_e{}_b{}".format(str(15), str(20))

if MODE == "TRAIN":
    model.summary()
    labels = np.array([(random.randint(-2,2), random.randint(-2,2)) for _ in range(60)], dtype=np.float32)
    model.fit(LLDs, labels, epochs = 15, batch_size=20, callbacks = [early_stop])
    model.save_weights(model_path)
    print("Model saved at {} ...".format(model_path))

###########
# Predict #
###########
elif MODE == "TEST":
    model.load_weights(model_path)
    idx = random.randint(0, len(LLDs))

    new_input = np.expand_dims(LLDs[idx], axis=0)

    prediction = model.predict(new_input)
    print("Affection Prediction:", prediction[0])