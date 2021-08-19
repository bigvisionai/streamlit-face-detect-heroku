import streamlit as st
import cv2
from PIL import Image
import numpy as np

st.title("OpenCV Deep Learning based Face Detection")

@st.cache(allow_output_mutation=True)
def load_model():
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
    return net
    

def detectFaceOpenCVDnn(net, frame, framework="caffe", conf_threshold=0.5):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]

    blob = cv2.dnn.blobFromImage(
        frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], False, False,
    )
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(
                frameOpencvDnn,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                int(round(frameHeight / 150)),
                8,
            )
    return frameOpencvDnn, bboxes


uploaded_file = st.file_uploader("Choose a file", type =['jpg','jpeg','jfif','png'])
net = load_model()
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    
    placeholders = st.beta_columns(2)
    placeholders[0].image(image)
    conf_threshold = st.slider("SET Confidence Threshold", min_value = 0.01, max_value = 1.0, step = .01, value=0.5)
    out_image,_ = detectFaceOpenCVDnn(net, image, conf_threshold=conf_threshold)
    
    placeholders[1].image(out_image)
