# Importing Libraries
import os
import streamlit as st
from PIL import Image
import torch
import pandas as pd

#Function to load the Uploaded Image
def load_image(image_file):
	img = Image.open(image_file)
	return img

#Function to load the Model
@st.cache(ttl=48*3600)
def load_model():
  model = torch.hub.load('yolov5','custom',path='best.pt',source='local', device='cpu',force_reload=True)
  return model

# Function to run on Streamlit
def main():

   # Setting the page title and favicon icon
   st.set_page_config(page_title='YOLO Classifier', page_icon='favicon.png')

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
        
        # Loading the model
        model = load_model()

        # Details of the image
        file_details = {"filename":image_file.name, "filetype":image_file.type,"filesize":image_file.size}
        st.write(file_details)
        img = load_image(image_file)
        st.image(img,width=640)
        
        #Loader for prrediction
        with st.spinner('Predicting...'):
            result=model(img,size=640)
            l= result.pandas().xyxy[0]['name']

        # Getting the count of detections
        d={}
        for i in l:
          d[i]=d.get(i,0)+1
        s=""
        for i in d:
          s+=f"{d[i]} {i}, "
        
        # Displaying Count on Streamlit
        st.success(s[:-2])

        # Displaying the prediction on image
        st.image(Image.fromarray(result.render()[0]))



if __name__ == '__main__':
    main()

