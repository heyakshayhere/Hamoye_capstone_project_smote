#Importing Libraries
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from PIL import Image
import streamlit as st

#Function to load the Uploaded Image
def load_image(image_file):
	img = Image.open(image_file)
	return img

# Function to get the pipelined model
@st.cache(ttl=48*3600)
def check():

    #Loading the pre-trained model using Keras
    lr = keras.models.load_model('weights.h5')

    #Prediction Pipeline
    class Preprocessor(BaseEstimator, TransformerMixin):
        def fit(self,img_object):
            return self
        
        def transform(self,img_object):
            img_array = image.img_to_array(img_object)
            expanded = (np.expand_dims(img_array,axis=0))
            return expanded

    class Predictor(BaseEstimator, TransformerMixin):
        def fit(self,img_array):
            return self
        
        def predict(self,img_array):
            probabilities = lr.predict(img_array)
            predicted_class = ['Buffalo', 'Elephant', 'Rhino', 'Zebra'][probabilities.argmax()]
            return predicted_class
    
    # Pipelined Model
    full_pipeline = Pipeline([('preprocessor',Preprocessor()),
                            ('predictor',Predictor())])
    return full_pipeline


#Function to predict the class
def output(full_pipeline,img):
    a=  img
    a= a.resize((256,256))
    predic = full_pipeline.predict(a)
    return(predic)

# Function to run on Streamlit
def main():
   # Setting the page title and favicon icon
   st.set_page_config(page_title='African Wildlife', page_icon='favicon.png')

   #Setting the page description
   st.title('African Wildlife Animal Classifier')
   st.subheader('Upload either Buffalo/Elephant/Rhino/Zebra image for prediction')

   #Image Uploader icon
   image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

   # Variable to store the prediction
   prediction = ''

   # Creating a Button for Prediction
   if st.button('Predict'):
     if image_file is not None:

         # Loader to depict the pre-trained model loading
         with st.spinner('Loading Image and Model...'):
            full_pipeline = check()
        
        # Details of the image
         file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
         st.write(file_details)
         img = load_image(image_file)
         st.image(img,width=256)

         #Loader for prediction
         with st.spinner('Predicting...'):
            prediction = output(full_pipeline,img)
         #Showing prediction
         st.success(prediction)

if __name__ == '__main__':
    main()
