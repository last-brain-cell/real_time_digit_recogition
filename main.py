import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
from keras.models import load_model
import cv2

# Create a canvas component
canvas_result = st_canvas(
    # fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color='#000000',
    background_color='#EEEEEE',
    background_image=None,
    update_streamlit=True,
    height=100,
    width=100,
    drawing_mode='freedraw',
    key="canvas",
)



# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
    num_classes = 10
    input_shape = (28, 28, 1)
    img = canvas_result.image_data

    resized_image = cv2.resize(canvas_result.image_data, (28, 28))

    # Convert the resized image to grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_RGBA2GRAY)

    # Reshape the grayscale image to match the model's input shape
    input_image = grayscale_image.reshape(-1, 28, 28, 1)

    # Normalize the pixel values to the range [0, 1]
    input_image = input_image.astype("float32") / 255.0

    model = load_model("modelx.h5")
    prediction = model.predict(input_image)

    pred = prediction.argmax(axis=1)
    conf = prediction[0,pred]
    print(pred)

    st.write(f"Prediction: {pred[0]}")
    st.write(f"Confidence: {round(conf[0]*100, 2)}%")