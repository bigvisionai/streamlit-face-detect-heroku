import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv_app

st.title("OpenCV Webinar Series")
st.header("Digit Recognizer using OpenCV and Streamlit")
st.write("Draw a digit over the canvas and click Predict")

# Create a canvas component
canvas_result = st_canvas(
    stroke_color="#fff",
    background_color="#000",
    height=250,
    width=250,
    key="canvas",
)

if st.button("predict"):
    classId, confidence = cv_app.predict(canvas_result)
    st.success("Predicted Class: {}, Confidence: {:.2f}%".format(classId, confidence*100.))

