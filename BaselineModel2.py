from __future__ import print_function
from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, Reshape, Dropout
from keras.models import Model
import sys
import csv
import numpy as np
from numpy import exp, array, random, dot
import pandas as pd
import nltk
import tensorflow as tf
import sklearn
from sklearn.model_selection import StratifiedKFold
import en_core_web_sm
from gensim.models import KeyedVectors
import keras
from keras import layers
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras_tqdm import TQDMNotebookCallback
from IPython import display
from keras.preprocessing.sequence import pad_sequences 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import fastText
import h5py

from embeddingMatrix import *


class BaselineModel2(keras.models.Model):
    
    def __init__(self, **kwargs):
        input_layer = layers.Input(
            shape=(MAX_SEQUENCE_LENGTH,),
            name='Input'
        )
        embedding_layer = layers.Embedding(len(vocabulary) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            name='Embedding',
                            trainable=False)(input_layer)

        conv1 = Conv1D(200, 2, activation='relu')(embedding_layer)
        conv2 = Conv1D(200, 3, activation='relu')(embedding_layer)
        conv3 = Conv1D(200, 4, activation='relu')(embedding_layer)
        
        pool1 = GlobalMaxPooling1D()(conv1)
        pool2 = GlobalMaxPooling1D()(conv2)
        pool3 = GlobalMaxPooling1D()(conv3)
        
        conc_layer = keras.layers.concatenate([pool3, pool2, pool1])
        drop1 = Dropout(0.5)(conc_layer)
        fc_layer1 = layers.Dense(units=30,name='FullyConnected')(drop1)
        drop2 = Dropout(0.5)(fc_layer1)
        #fc_layer2 = layers.Flatten(name="Flatten")(drop2)
        predictions = Dense(6, activation='softmax')(drop2)
        #super().__init__(inputs=[input_layer], outputs=predictions)
        super().__init__(**kwargs)
        
    def compile(self):
        return super().compile(
            optimizer=keras.optimizers.Adam(lr=0.001),
            loss='mse'
        )