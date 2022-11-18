import streamlit as st  
import tensorflow as tf
import cv2
import keras
import numpy as np
from PIL import Image, ImageOps

st.set_option('deprecation.showfileUploaderEncoding',
              False)


@st.cache(
    allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model(r'/Users/kirubhakaran/PycharmProjects/web application/SignLanguageClasifier.h5')
    return model


model = load_model()

st.markdown("<h1 style='text-align: center; color: Black;'>Sign Language Classifier</h1>", unsafe_allow_html=True)
st.markdown(
    "<h3 style='text-align: center; color: Black;'>All you have to do is Upload the Sign Language Image and the model will do the rest!</h3>",
    unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: Black;'>Submission for IBM NALAIYATHIRAN </h4>", unsafe_allow_html=True)
st.sidebar.header("What is this Project about?")
st.sidebar.text("It is a Deep learning solution to detection of Sign Language Prediction of Indian Sign Language from 0-9 and A-Z.")

file = st.file_uploader("Please upload your MRI Scan", type=["jpg", "png"])  # accepting the image input from the user


def import_and_predict(img,
                       model):
    img = edge_detection(img)
    img = cv2.resize(img, (64, 64))
    prediction = model.predict(img)
    return prediction


if file is None:
    st.markdown("<h5 style='text-align: center; color: Black;'>Please Upload a Sign Language Image</h5>",
                unsafe_allow_html=True)
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    class_names = ['1','2','3','4','5','6','7','8','9','0','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R',
                   'S','T','U','V','W','X','Y','Z']

    string = "The predicted sign language is : " + class_names[np.argmax(predictions)]
    st.success(string)
    # st.success(predictions)