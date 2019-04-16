from __future__ import print_function
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

dfTrain = pd.read_csv("tweetTrainingContextFilter2.csv")
df3= dfTrain[dfTrain.label == 3]
df3=df3[:3697]
df1=dfTrain[dfTrain.label == 1]
df5=dfTrain[dfTrain.label == 5]
df5=df5[:3697]
dfTrain = df1.append(df3, ignore_index=True)
dfTrain= dfTrain.append(df5, ignore_index=True)
#CHANGED sequence length from 100 to 50
MAX_SEQUENCE_LENGTH = 40
#EMBEDDING_DIM = wikiModel.dim
#dimension of wikiModel
EMBEDDING_DIM = 300
#print(EMBEDDING_DIM)

def wordEmbedding(dfTrain):
    #Consider creating vocabulary from wiki
    vocabulary = dict()
    inverse_vocabulary = ['PADDING']
    sequences = []
    for text in dfTrain.body:
        text = text.split()
        text_sequence = []
        for word in text:
            if word not in vocabulary:
                vocabulary[word] = len(inverse_vocabulary)
                text_sequence.append(len(inverse_vocabulary))
                inverse_vocabulary.append(word)
            else:
                text_sequence.append(vocabulary[word])
        sequences.append(text_sequence)
    print("%d unique tokens in the vocabulary" %len(vocabulary))

    bodies_seq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    return vocabulary

    #consider doing more text cleaning
import re
def clean_text(text):
    '''Clean text by removing unnecessary characters and altering the format of words.'''

    text = text.lower()
    
    text = re.sub(r"\n", "",  text)
    text = re.sub(r"[-()]", "", text)
    text = re.sub(r"\.", " .", text)
    text = re.sub(r"\!", " !", text)
    text = re.sub(r"\?", " ?", text)
    text = re.sub(r"\,", " ,", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"ohh", "oh", text)
    text = re.sub(r"ohhh", "oh", text)
    text = re.sub(r"ohhhh", "oh", text)
    text = re.sub(r"ohhhhh", "oh", text)
    text = re.sub(r"ohhhhhh", "oh", text)
    text = re.sub(r"ahh", "ah", text)
    
    return text

bodies2 = []
for line in dfTrain.body:
    bodies2.append(clean_text(line))
bodies=bodies2

#gets unique tokens after cleaning text
vocabulary = dict()
inverse_vocabulary = ['PADDING']
sequences = []
for text in bodies:
    text = text.split()
    text_sequence = []
    for word in text:
        if word not in vocabulary:
            vocabulary[word] = len(inverse_vocabulary)
            text_sequence.append(len(inverse_vocabulary))
            inverse_vocabulary.append(word)
        else:
            text_sequence.append(vocabulary[word])
    sequences.append(text_sequence)
print("%d unique tokens in the vocabulary" %len(vocabulary))

bodies_seq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
