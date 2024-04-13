import PIL
import streamlit as st
from pathlib import Path
from ImageDetector import ImageDetector
from VideoDetector import VideoDetector
from YouTubeDetector import YouTubeDetector
from WebcamDetector import WebcamDetector
import settings
import helper
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Setting page layout
st.set_page_config(
    page_title="Object Detection And Tracking using YOLOv9",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Main page heading
st.title("Object Detection And Tracking using YOLOv9")

# Sidebar
st.sidebar.header("ML Model Config")
model_type = st.sidebar.radio("Select Task", ['BEST','TBM_SAFETY'])
confidence = float(st.sidebar.slider("Select Model Confidence", 30, 100, 40)) / 100

# Selecting Detection Or Segmentation
if model_type == "BEST":
    model_path = Path(settings.BEST_MODEL)
elif model_type == "TBM_SAFETY":
    model_path = Path(settings.TMB_SAFETY_MODEL)

# Load Pre-trained ML Model
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Error loading model. Check the specified path: {model_path}")
    st.error(ex)

# Sidebar
st.sidebar.header("Data Config")
source_radio = st.sidebar.radio("Select Source", ["Image", "Video", "Youtube", "RTSP", "Webcam"])


if source_radio == "Image":
    image_detector = ImageDetector(model, confidence)
    image_detector.detect()
elif source_radio == "Youtube":
    youtube_detector = YouTubeDetector(model, confidence)
    youtube_detector.detect()
elif source_radio == "Video":
    video_detector = VideoDetector(model, confidence)
    video_detector.detect()
elif source_radio == "Webcam":
    webcam_detector = WebcamDetector(model, confidence)
    webcam_detector.detect()
elif source_radio in [settings.RTSP]:
    helper.play_rtsp_stream(confidence, model)

