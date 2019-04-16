import numpy as np
import keras.models
import h5py
from keras.models import model_from_json
from scipy.misc import imread, imresize,imshow

from keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, Reshape, Dropout
from keras.models import Model
from keras.utils import CustomObjectScope
from keras import layers
from keras import backend as K
from keras.preprocessing.sequence import pad_sequences 
from keras.models import Sequential
from keras.layers import Dense
from BaselineModel2 import *


model_file = "model2.json"

weight_file = "model2.h5"
with open(model_file, 'r') as jfile:
    model = model_from_json(jfile.read(), custom_objects={'BaselineModel2': BaselineModel2()})
#with CustomObjectScope(custom_objects={'BaselineModel2': BaselineModel2}):
#    model = load_model('model2.h5')



#load woeights into new model

model.load_weights(weight_file)

print("Loaded Model from disk")
#json_file = open("model2.json",'r')
#loaded_model_json = json_file.read()
#json_file.close()
#loaded_model = model_from_json(loaded_model_json, custom_objects={'BaselineModel2': BaselineModel2})
#loaded_model = model_from_json(loaded_model_json)
#model = load_model('model2.h5', custom_objects={'BaselineModel2': BaselineModel2})
#loaded_model = model.load_weights("model2.h5")
#load woeights into new model

#compile and evaluate loaded model
model.compile()

#loss,accuracy = model.evaluate(X_test,y_test)
#print('loss:', loss)
#print('accuracy:', accuracy)
#x = imread('output.png',mode='L')
#x = np.invert(x)
#x = imresize(x,(28,28))
#imshow(x)
#x = x.reshape(1,28,28,1)

#out = loaded_model.predict(x)
#print(out)
#print(np.argmax(out,axis=1))
