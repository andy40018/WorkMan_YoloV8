import PIL
import settings
import streamlit as st
import helper

class ImageDetector:
    def __init__(self, model, accuracy):
        self.model = model
        self.accuracy = accuracy
    def detect(self):
        image_process = None
        label_count = 0
        
        # uploader image
        source_image = st.sidebar.file_uploader(
            "Upload an image", type=("jpg", "jpeg", "png", "bmp", "webp")
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            image_process = helper.display_uploaded_image(source_image)
        
        if st.sidebar.button("Detect Objects"):
            if image_process is None:
                st.warning("Please upload an image first.")
                return
            
            res = self.model.predict(image_process, conf=self.accuracy)
            boxes = res[0].boxes
            res_plotted = res[0].plot()[:,:,::-1]

            with col2:
                st.image(res_plotted, caption='Detected Image', use_column_width=True)
                
                try:
                    with st.expander("Detection Results"):
                        if not boxes:
                            st.write("No objects detected")
                        else:
                            # 專門處理物件檢測的結果
                            detected_objects_summary_list, label_count = helper.process_detection_results(boxes, image_process, self.model, label_count)
                            helper.sum_detections(detected_objects_summary_list, self.model)
                except Exception as ex:
                    st.write("An error occurred while processing the detection results : " + str(ex))    
    # def detect(self):
    #     image_process = None
    #     label_count = 0
    #     # uploader image
    #     source_image = st.sidebar.file_uploader(
    #     "Upload an image", type=("jpg", "jpeg", "png", "bmp", "webp")
    #     )
    #     col1, col2 = st.columns(2)
    #     with col1:
    #         try:
    #             if source_image is not None:
    #                 image_process = PIL.Image.open(source_image)
    #                 st.image(image_process,caption="Uploaded Image", use_column_width=True)
    #             else:
    #                 default_image = PIL.Image.open(settings.DEFAULT_IMAGE)
    #                 st.image(default_image,caption="Default Image", use_column_width=True)
    #                 image_process = default_image
    #         except Exception as ex:
    #             st.error(f"Error loading image")
    #             st.error(ex)
    #     if st.sidebar.button("Detect Objects"):
    #         detected_objects_summary_list = []
    #         res = self.model.predict(image_process, conf=self.accuracy)
    #         boxes = res[0].boxes
    #         res_plotted = res[0].plot()[:,:,::-1]
    #         detected_objects_summary_list.extend(res[0].boxes.cls)

    #         with col2:
    #             st.image(res_plotted, caption='Detected Image', use_column_width=True)
    #             try:
    #                 with st.expander("Detection Results"):
    #                     if not boxes:
    #                         st.write("No objects detected")
    #                     else:
    #                         for person_box in boxes:
    #                             if self.model.names[int(person_box.cls)] == 'Person':
    #                                 person_x1, person_y1, person_x2, person_y2 = map(int, person_box.xyxy[0])
                                    
    #                                 # Check if 'Hardhat' is detected within 'Person' box
    #                                 hardhat_detected = False
    #                                 for hardhat_box in boxes:
    #                                     if self.model.names[int(hardhat_box.cls)] == 'Hardhat':
    #                                         hardhat_x1, hardhat_y1, hardhat_x2, hardhat_y2 = map(int, hardhat_box.xyxy[0])
    #                                         if person_x1 <= hardhat_x1 <= person_x2 and person_y1 <= hardhat_y1 <= person_y2:
    #                                             hardhat_detected = True
    #                                             break
                                    
    #                                 # Crop and save the 'Person' object if 'Hardhat' is not detected
    #                                 if not hardhat_detected:
    #                                     cropped_image = image_process.crop((person_x1, person_y1, person_x2, person_y2))
    #                                     helper.saveBoxesImage(label_count, self.model.names[int(person_box.cls)], cropped_image)  
    #                                     label_count += 1
    #             except Exception as ex:
    #                 st.write("An error occurred while processing the detection results : " + str(ex))
    #         if boxes:
    #             helper.sum_detections(detected_objects_summary_list, self.model)              
