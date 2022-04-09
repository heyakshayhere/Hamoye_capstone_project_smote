# Hamoye : Team SMOTE

### Object detection using TensorFlow and YOLOv5

## **Approach:**

### Dataset used: [African Wildlife](https://www.kaggle.com/datasets/biancaferreira/african-wildlife)

The original dataset was YOLO specific so our team split into two parts:

## 1. Team Tensorflow

### [Deployed Model Link](https://share.streamlit.io/vermaayush680/streamlit/main/predictions.py)

  ![Tensorflow Logo](https://www.logolynx.com/images/logolynx/s_29/29e59a2b11f45a2dbbdc5f034e2a5a0e.png)
  
  **WorkFlow:**
  * Data Collection
  * Data Cleaning
  * Data Augmentation
  * Modelling
  * Evaluation
  * Deployment
  
  For the tensorflow models, we collected animal images of animlas through data scrapping as our YOLO dataset had images with multiple animals in them reducing the
  accuracy of the model.
  
  The team used the following models:
  * EfficientNet
  * Inception V3
  * MobileNet V2
  * NASNetLarge
  * Resnet152
  * Resnet50
  * VGG16
  
  And finally the **EfficientNetB7** model was selected for deployment.
  
  

## 2. Team YOLO

### [Deployed Model Link](https://share.streamlit.io/av84190875/yolo-africa/main/prediction.py)

![YOLO](https://user-images.githubusercontent.com/42459801/162487411-8d82084e-2a56-4d51-8d97-cec60b715f8b.png)

  For the YOLO model, we used the original kaggle dataset and used **YOLOv5x6** for the final modelling.

**WorkFlow:**
* Modelling
* Evaluation
* Deployment



# YOLO Video Tests 

https://user-images.githubusercontent.com/42459801/162434193-f4e47abd-47ca-4a79-ae71-8dabbfa2ce43.mp4


https://user-images.githubusercontent.com/42459801/162434231-45665cf6-648c-4c27-a302-b74492ad2804.mp4


https://user-images.githubusercontent.com/42459801/162559791-059edfa1-518e-4207-9af8-ea5933c43463.mp4


