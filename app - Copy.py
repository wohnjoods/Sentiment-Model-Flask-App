#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html
#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.
#IMPORTS:
import csv
import tensorflow as tf
#import sklearn
#from sklearn.model_selection import StratifiedKFold
#import en_core_web_sm
#from gensim.models import KeyedVectors
#from IPython import display
#from keras.preprocessing.sequence import pad_sequences 
import fastText
import h5py
#requests are objects that flask handles (get set post, etc)
from flask import Flask, render_template,request
#scientific computing library for saving, reading, and resizing images
#from scipy.misc import imsave, imread, imresize
#for matrix math
import numpy as np
#for importing our keras model
from keras import models
#for regular expressions, saves time dealing with string data
import re
#system level operations (like loading files)
import sys 
#for reading operating system data
import os
import pandas as pd
#tell our app where our saved model is (don't think this is used)
sys.path.append(os.path.abspath("./model2"))
from load import * 
from model2 import *
#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
global model, graph
graph = tf.get_default_graph()
model= loadModel2()
#initialize these variables
#model, graph = init()
#vocab = pd.read_csv("vocabModel-299-2000-120.csv")
EMBEDDING_DIM = 300
modelPath="model2.json"
weightPath="model2.h5"
from keras.models import load_model



 
@app.route("/", methods=['POST','GET'])
def index():
    return render_template('index - Copy.html')


@app.route("/predict", methods=["GET",'POST'])
def predict(): 

    textInput = request.form['textInput']
    
    x= format_input(textInput)
    with graph.as_default():

		#perform the prediction
            
	    output = model.predict(x)

	    output= np.asarray([np.argmax(i) for i in output])

    output=convertOutput(output)
    return render_template('index - Copy.html', input=request.form['textInput'],output=output)
    




if __name__ == "__main__":
    app.run(debug=True)
    