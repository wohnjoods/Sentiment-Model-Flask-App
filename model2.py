import csv
import keras
import tensorflow as tf
from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, Reshape, Dropout
from keras.models import Model
from keras import models
from keras import layers
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
import pandas as pd
import re
with open('vocabModel-299-2000-120.csv') as f:
    vocabulary = dict(filter(None, csv.reader(f)))
EMBEDDING_DIM = 300
modelPath="model2.json"
weightPath="model2.h5"
MAX_SEQUENCE_LENGTH = 40
text_seq=["I", "Hate"]

global model, graph
#initialize these variables
#model, graph = init()
#print(len(vocabulary))
embedding_matrix = np.zeros((len(vocabulary)+1, EMBEDDING_DIM))

def load_model():

    # load json and create model
    model = model2()
    model.compile()
    model.load_weights("model2.h5")
    
    return model2

def format_input(inputText):
    inputText = clean_text(inputText)
    #convert to sequence
    #inverse_vocabulary = ['PADDING']
    inverse_vocabulary = []
    sequences = []
    #for text in inputText:
    inputText = inputText.split()
    text_sequence = []
    for word in inputText:
        if word not in vocabulary:
            vocabulary[word] = len(inverse_vocabulary)
            text_sequence.append(len(inverse_vocabulary))
            inverse_vocabulary.append(word)
        else:
            text_sequence.append(vocabulary[word])
    sequences.append(text_sequence)

    bodies_seq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
    
    return bodies_seq
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

class model2(keras.models.Model):
    
    def __init__(self):
        input_layer = layers.Input(
            shape=(MAX_SEQUENCE_LENGTH,),
            name='Input'
        )
        embedding_layer = layers.Embedding(len(vocabulary), EMBEDDING_DIM, weights=[embedding_matrix], input_length=MAX_SEQUENCE_LENGTH, name='Embedding', trainable=False)(input_layer)

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
        super().__init__(inputs=[input_layer], outputs=predictions)
        
    def compile(self):
        return super().compile(
            optimizer=keras.optimizers.Adam(lr=0.001),
            loss='mse'
        )

def loadModel2(): 
    textInput="blah"
    #textInput = request.form['textInput']
    textInput = format_input(textInput)
    #model2 = load_model()
    model = model2()
    
    #graph = tf.get_default_graph()
    model.compile()
    
    model.load_weights("model2.h5")
    #model._make_predict_function()
    
    #return render_template('index - Copy.html', input=request.form['textInput'])
    return model

def convertOutput(array):
    output=array[0]
    if output == 1:
        output="\U0001F621"
    elif output==3:
        output="\U0001F610"
    else:
        output="\U0001F604"
    return output