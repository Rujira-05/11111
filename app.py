# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 13:52:05 2025
@author: LAB
"""

import streamlit as st
import numpy as np
from PIL import Image
import pickle
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set title of the application
st.title("Image Classification with MobileNetV2 by Rujira Moonha")

# File upload
uploaded_file = st.file_uploader("Upload image:", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image on screen
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)


    # Preprocessing
    img = img.resize((224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Display prediction
    preds = model.predict(x)
    top_preds = decode_predictions(preds, top=3)[0]

    # Display predictions
    st.subheader("Predictions:")
    for i, pred in enumerate(top_preds):
        st.write(f"{i+1}. **{pred[1]}** - {round(pred[2]*100, 2)}%")
