import cv2
import streamlit as st
from pytube import YouTube
import settings
import helper

class YouTubeDetector:
    def __init__(self, model, accuracy):
        self.model = model
        self.accuracy = accuracy

    def detect(self):
        source_youtube = st.sidebar.text_input(
            "YouTube Video url", settings.DEFAULT_URL
        )

        is_display_tracker, tracker = helper.display_tracker_options()
        detected_objects_summary_list = []
        if st.sidebar.button("Detect Objects"):
            try:
                yt = YouTube(source_youtube)
                stream = yt.streams.filter(file_extension="mp4", res=720).first()
                vid_cap = cv2.VideoCapture(stream.url)

                st_frame = st.empty()
                while vid_cap.isOpened():
                    success, image = vid_cap.read()
                    if success:
                        res = helper.display_frames(
                            self.model,
                            self.accuracy,
                            st_frame,
                            image,
                            is_display_tracker,
                            tracker,
                        )
                        detected_objects_summary_list.extend(res[0].boxes.cls)
                    else:
                        vid_cap.release()
                        helper.sum_detections(detected_objects_summary_list, self.model)
                        break
            except Exception as e:
                st.sidebar.error("Error loading video: " + str(e))
