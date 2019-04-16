#our web app framework!

#you could also generate a skeleton from scratch via
#http://flask-appbuilder.readthedocs.io/en/latest/installation.html
#Generating HTML from within Python is not fun, and actually pretty cumbersome because you have to do the
#HTML escaping on your own to keep the application secure. Because of that Flask configures the Jinja2 template engine 
#for you automatically.

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
#tell our app where our saved model is
sys.path.append(os.path.abspath("./model2"))
from load import * 

#initalize our flask app
app = Flask(__name__)
#global vars for easy reusability
global model, graph
#initialize these variables
model, graph = init()

print("check1")
#def convertText(textData):

	#textStr = re.search(r'base64,(.*)',imgData1).group(1)

	#print(imgstr)

	#with open('output.png','wb') as output:

	#	output.write(imgstr.decode('base64'))


@app.route('/')
def index():

	#initModel()

	#render out pre-built HTML file right on the index page
	
	return render_template("index.html")

print("check2")
@app.route('/predict',methods=['POST'])
def predict():
	return "You said: " + request.form['text']

	#whenever the predict method is called, we're going

	#to input the text into the model

	#perform inference, and return the classification

	#get data for string

#	textData = request.get_data()

	#encode it into a suitable format for model

	#convertText(textData)

	print ("text converted for model")

	#read the text into memory

	#x = imread('output.txt',mode='L')


	#imshow(x)

	#convert to a 2D tensor to feed into our model

	#x = x.reshape(1,40)?

	print ("debug2")

	#in our computation graph

	#with graph.as_default():

		#perform the prediction

	#	out = model.predict(x)

	#	print(out)

	#	print(np.argmax(out,axis=1))

	#	print ("debug3")

		#convert the response to a string

	#	response = np.array_str(np.argmax(out,axis=1))

	#	return response	

print("check3")
if __name__ == "__main__":

	#decide what port to run the app in

	#port = int(os.environ.get('PORT', 5000))

	#run the app locally on the givn port

	#app.run(host='0.0.0.0', port=port)

	#optional if we want to run in debugging mode

	app.run(debug=True)

	

	
