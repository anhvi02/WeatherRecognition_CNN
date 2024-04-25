import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import requests
from time import sleep

### PAGE SETTING ---------------------------------------------->
st.set_page_config(
    page_title="Weather Recognition",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state = 'auto'
)



# side bar
def sidebar_bg(img_url):
    side_bg_ext = 'jpg'  # Assuming the image format is JPG
    # Retrieve image data from URL
    response = st.cache_resource(requests.get)(img_url, stream=True)
    img_data = response.content
    # Encode image data as base64
    encoded_data = base64.b64encode(img_data).decode()
    # Apply background image style to sidebar
    st.markdown(
        f"""
        <style>
        [data-testid="stSidebar"] > div:first-child {{
            background: url(data:image/{side_bg_ext};base64,{encoded_data});
            background-size: cover; 
        }}
        .title-message {{
            color: white;
            font-weight: bold;
            font-size: 50px;
            padding-top: 50%
        }}
        .dev-message {{
            color: white;
            font-weight: bold;
        }}
        </style>
        """,
        unsafe_allow_html=True)

img_url = "https://i.pinimg.com/564x/da/db/c2/dadbc236d835649deb617eb3852df54a.jpg"
sidebar_bg(img_url)

with st.sidebar:
    message_html = """
    <div class="title-message">Weather Recognition Model</div>
    <h3 class="dev-message">Developed by Anh Vi Pham</h5>
    """
    st.sidebar.write(message_html, unsafe_allow_html=True)



### PREPARE MODEL ---------------------------------------------->

# load model
@st.cache_resource
def load_keras_model():
    model = load_model('model_weather_anhvi02.keras')
    return model
with st.spinner('Model is being loaded..'):
    model = load_keras_model()



filler1, content, filler = st.columns([1,8,1])
with content:
    st.subheader("Drop your image here")
    ### GET DATA ---------------------------------------------->
    # file uploader 
    file_upload = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], )
    # function to extract data from img
    def extract_img(file_image, resize_size, img_channel=3):
        if file_image is not None:
            # Read the content of the uploaded file as bytes
            content = file_image.read()
            # Convert bytes to a TensorFlow tensor
            img = tf.image.decode_jpeg(content, channels=img_channel)
            img = tf.image.resize(img, resize_size)
            # Convert pixel values to the range [0, 1]
            img = tf.cast(img, tf.float32) / 255.0
            # Expand dimensions to add a batch dimension
            img = tf.expand_dims(img, axis=0)
            return img
        else:
            return None
        
        
    if file_upload is not None:
        # specify size to resize image
        resize_size = (200, 200)
        # Extract image data
        img_extracted = extract_img(file_upload, resize_size, 3)

    ### PERFORM PREDICTION ---------------------------------------------->

    # dictionrary of label with their index
    label_map = {0: 'FOG' , 1: 'RAIN', 2: 'RIME', 3: 'SANDSTORM', 4: 'SNOW'}

    # Function to interpret predictions
    def interpret_prediction(predictions):
        # get index of the label having the highest probability
        label_index = int(np.argmax(predictions, axis=1)[0])
        # get the label from the label_map dictionary using labe_index
        prediction_label = label_map[label_index]
        return prediction_label

    # Check if image data is extracted and display predict button
    if file_upload is not None:
        with st.spinner('Model making prediction...'):
            # Make predictions
            prediction = model.predict(img_extracted)
            
            # interpret
            label_prediction = interpret_prediction(prediction)
            # Display prediction result
            st.info(f'{label_prediction}')
        # toast
        sleep(1)
        st.toast('Prediction Successful!', icon='🎉')
        
    # display image
    if file_upload is not None:
        st.write("")
        empty1, display_image, empty2 = st.columns([2,6,2])
        with display_image:
            image = Image.open(file_upload)
            st.image(image, use_column_width=True)