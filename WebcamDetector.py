import cv2
import streamlit as st
import helper
from PIL import Image
import numpy as np



class WebcamDetector:
    def __init__(self, model, accuracy):
        self.model = model
        self.accuracy = accuracy
        self.quit_flag = False 
        
    def detect(self):
        is_display_tracker, tracker = helper.display_tracker_options()
        label_count = 0

        notify_on = False

        # Initialize a global variable to store the state
        if 'detected_objects_summary_list' not in st.session_state:
            st.session_state.detected_objects_summary_list = []

        IsNotify = st.radio("Turn On LINE  Notify", ("On","Off"))
        if st.sidebar.button("Turn On Webcam"):        
                try:
                    vid_cap = cv2.VideoCapture(0)
                    
                    st_frame = st.empty()
                    while vid_cap.isOpened() and not self.quit_flag:
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

                            # Check if detection notification is enabled
                            # if st.sidebar.button("Turn On LINE  Notify",key="turn_on_notify_button"):
                            
                            IsNotify_On = True if IsNotify == "On" else False
                            if IsNotify_On:
                                notify_on = True

                            if notify_on:
                                boxes = res[0].boxes
                                image_pil = Image.fromarray(image)
                                # Process detection results

                                # 待修改使用多目標物件識別，目前先以每次發送
                                detected_objects_summary_list, label_count = helper.process_detection_results_TimeScan(
                                    boxes, image_pil, self.model, label_count
                                )
                                # helper.sum_detections(detected_objects_summary_list,self.model)

                            st.session_state.detected_objects_summary_list.extend((res[0].boxes.cls).tolist())
                        else:
                            vid_cap.release()
                except Exception as e:
                    st.sidebar.error("Error loading video: " + str(e))
                    print("Error loading video: " + str(e))
        if st.sidebar.button('Quit Webcam'):
            self.quit_flag = True
            helper.sum_detections(st.session_state.detected_objects_summary_list, self.model)
            st.session_state.detected_objects_summary_list = []

           

