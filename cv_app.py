import streamlit as st
import cv2
import numpy as np

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

@st.cache(allow_output_mutation=True)
def load_model():
    net = cv2.dnn.readNetFromONNX('model.onnx')
    return net

def predict(canvas_result):
    net = load_model()

    img = cv2.cvtColor(np.uint8(canvas_result.image_data), cv2.COLOR_RGBA2GRAY)

    # Create a 4D blob from image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (28, 28))
    # Run a model
    net.setInput(blob)

    out = net.forward()

    # Get a class with a highest score
    out = softmax(out.flatten())
    classId = np.argmax(out)
    confidence = out[classId]

    return classId, confidence
