from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube
import settings
import requests
import os
import PIL
from datetime import datetime
import time

# Line 通知
url = "https://notify-api.line.me/api/notify"
token = "Ui0pBO2uk1OLdhMxN08vVvh0rAlo7STUdB4U8Bx9UsH"
def lineNotify(msg,image_path=None):
    headers = {
        'Authorization': 'Bearer ' + token
    }

    payload = {'message': msg}
    files = {'imageFile': open(image_path, 'rb')} if image_path else None

    r = requests.post(url, headers=headers, data=payload, files=files)

# 儲存成圖片再發給推播
def saveBoxesImage(index,waring,save_image, save_path=settings.RESULTERR_DIR):
    """
    Saves segmented objects from the image based on detected boxes.

    Parameters:
        boxes (list): List of detected boxes.
        save_image (PIL.Image): The original image.
        save_path (str): Directory path to save segmented objects.

    Returns:
        None
    """
    now = datetime.now()
    year = now.year
    month = now.month
    day = now.day
    folder_name = f"{save_path}/{year}{month:02d}{day:02d}"

    # Create directory if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    save_image.save(f"{folder_name}/{waring}_{index}.png")
    lineNotify("warning_" + f"{index}",image_path=f"{folder_name}/{waring}_{index}.png")

# 負責展示上傳的圖像或者默認圖像
def display_uploaded_image(image):
    """
    Display the uploaded image or default image.
    """
    try:
        if image is not None:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        else:
            default_image = PIL.Image.open(settings.DEFAULT_IMAGE)
            st.image(default_image, caption="Default Image", use_column_width=True)
            return default_image
    except Exception as ex:
        st.error(f"Error loading image")
        st.error(ex)
        return None

# 專門處理物件檢測的結果
def process_detection_results(boxes, image_process, model, label_count):
    """
    Process detection results to save cropped images.
    """
    detected_objects_summary_list = []
    for person_box in boxes:
        if model.names[int(person_box.cls)] == 'Person':
            person_x1, person_y1, person_x2, person_y2 = map(int, person_box.xyxy[0])
            
            hardhat_detected = False
            for hardhat_box in boxes:
                if model.names[int(hardhat_box.cls)] == 'Hardhat':
                    hardhat_x1, hardhat_y1, hardhat_x2, hardhat_y2 = map(int, hardhat_box.xyxy[0])
                    if person_x1 <= hardhat_x1 <= person_x2 and person_y1 <= hardhat_y1 <= person_y2:
                        hardhat_detected = True
                        break
            
            if not hardhat_detected:
                cropped_image = image_process.crop((person_x1, person_y1, person_x2, person_y2))
                saveBoxesImage(label_count, model.names[int(person_box.cls)], cropped_image)  
                label_count += 1

            detected_objects_summary_list.extend(person_box.cls)
    
    return detected_objects_summary_list, label_count

person_no_hardhat_time = {}  # Dictionary to store start time for each person without a hardhat
# 檢測時間內持續發生時觸發
def process_detection_results_TimeScan(boxes, image_process, model, label_count, time_detection=False, time_limit=2):
    """
    Process detection results to save cropped images with optional time detection.
    
    Args:
    - boxes (list): Detected boxes from the model.
    - image_process (PIL.Image): Processed image from the uploader.
    - model (object): Detection model.
    - label_count (int): Counter for saving images.
    - time_detection (bool, optional): Whether to enable time detection. Defaults to False.
    - time_limit (int, optional): Time limit in seconds. Defaults to 5.
    
    Returns:
    - detected_objects_summary_list (list): List of detected object classes.
    - label_count (int): Updated counter for saving images.
    """
    detected_objects_summary_list = []
    
    for person_box in boxes:
        if model.names[int(person_box.cls)] == 'Person':
            person_x1, person_y1, person_x2, person_y2 = map(int, person_box.xyxy[0])
            
            hardhat_detected = False
            for hardhat_box in boxes:
                if model.names[int(hardhat_box.cls)] == 'Hardhat':
                    hardhat_x1, hardhat_y1, hardhat_x2, hardhat_y2 = map(int, hardhat_box.xyxy[0])
                    if person_x1 <= hardhat_x1 <= person_x2 and person_y1 <= hardhat_y1 <= person_y2:
                        hardhat_detected = True
                        
                        if time_detection and person_box.cls not in person_no_hardhat_time:
                            person_no_hardhat_time[person_box.cls] = time.time()
                        break
            
            if not hardhat_detected:
                if time_detection:
                    if person_box.cls in person_no_hardhat_time:
                        elapsed_time = int(time.time() - person_no_hardhat_time[person_box.cls])  # Convert to int for accurate comparison
                        if elapsed_time >= time_limit:
                            cropped_image = image_process.crop((person_x1, person_y1, person_x2, person_y2))
                            saveBoxesImage(label_count, model.names[int(person_box.cls)], cropped_image)  
                            label_count += 1
                        else:
                            continue
                    else:
                        person_no_hardhat_time[person_box.cls] = time.time()
                else:
                    cropped_image = image_process.crop((person_x1, person_y1, person_x2, person_y2))
                    saveBoxesImage(label_count, model.names[int(person_box.cls)], cropped_image)  
                    label_count += 1

            detected_objects_summary_list.extend(person_box.cls)
    
    return detected_objects_summary_list, label_count

    # for person_box in boxes:
    #     if model.names[int(person_box.cls)] == 'Person':
    #         person_x1, person_y1, person_x2, person_y2 = map(int, person_box.xyxy[0])
            
    #         hardhat_detected = False
    #         for hardhat_box in boxes:
    #             if model.names[int(hardhat_box.cls)] == 'Hardhat':
    #                 hardhat_x1, hardhat_y1, hardhat_x2, hardhat_y2 = map(int, hardhat_box.xyxy[0])
    #                 if person_x1 <= hardhat_x1 <= person_x2 and person_y1 <= hardhat_y1 <= person_y2:
    #                     hardhat_detected = True
                        
    #                     if time_detection and person_box.cls not in person_no_hardhat_time:
    #                         person_no_hardhat_time[person_box.cls] = time.time()
    #                     break
            
    #         if not hardhat_detected:
    #             # if time_detection:
    #             #     if person_box.cls in person_no_hardhat_time:
    #             #         elapsed_time = time.time() - person_no_hardhat_time[person_box.cls]
    #             #         if elapsed_time >= time_limit:
    #             #             cropped_image = image_process.crop((person_x1, person_y1, person_x2, person_y2))
    #             #             saveBoxesImage(label_count, model.names[int(person_box.cls)], cropped_image)  
    #             #             label_count += 1
    #             #         else:
    #             #             continue
    #             #     else:
    #             #         person_no_hardhat_time[person_box.cls] = time.time()
    #             if time_detection:
    #                 if person_box.cls in person_no_hardhat_time:
    #                     elapsed_time = int(time.time() - person_no_hardhat_time[person_box.cls])  # Convert to int for accurate comparison
    #                     if elapsed_time >= time_limit:
    #                         cropped_image = image_process.crop((person_x1, person_y1, person_x2, person_y2))
    #                         saveBoxesImage(label_count, model.names[int(person_box.cls)], cropped_image)  
    #                         label_count += 1
    #                     else:
    #                         continue
    #                 else:
    #                     person_no_hardhat_time[person_box.cls] = time.time()

    #             else:
    #                 cropped_image = image_process.crop((person_x1, person_y1, person_x2, person_y2))
    #                 saveBoxesImage(label_count, model.names[int(person_box.cls)], cropped_image)  
    #                 label_count += 1

    #         detected_objects_summary_list.extend(person_box.cls)
    
    # return detected_objects_summary_list, label_count
    

def load_model(model_path):
    """
    Loads a YOLO object detection model from the specified model_path.

    Parameters:
        model_path (str): The path to the YOLO model file.

    Returns:
        A YOLO object detection model.
    """
    model = YOLO(model_path)
    return model


def display_tracker_options():
    """
    Displays options for enabling object tracking in the Streamlit app.

    Returns:
        Tuple (bool, str): A tuple containing a boolean flag for displaying the tracker and the selected tracker type.
    """
    display_tracker = st.radio("Display Tracker", ("No","Yes"))
    is_display_tracker = True if display_tracker == "Yes" else False
    if display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


def display_frames(
    model, acc, st_frame, image, is_display_tracker=None, tracker_type=None
):
    """
    Displays detectes objects from a video stream.

    Parameters:
        model (YOLO): A YOLO object detection model.
        acc (float): The model's confidence threshold.
        st_frame (streamlit.Streamlit): A Streamlit frame object.
        image (PIL.Image.Image): A frame from a video stream.
        is_display_tracker (bool): Whether or not to display a tracker.
        tracker_type (str): The type of tracker to display.

    Returns:
        None
    """

    image = cv2.resize(image, (720, int(720 * (9 / 16))))
    if is_display_tracker:
        res = model.track(image, conf=acc, persist=True, tracker=tracker_type)
    else:
        res = model.predict(image, conf=acc)

    res_plot = res[0].plot()
    st_frame.image(
        res_plot,
        caption="Detected Video",
        channels="BGR",
        use_column_width=True,
    )
    return res


def sum_detections(detected_objects_summary_list, model):
    """
    Summarizes detected objects from a list and displays the summary in a Streamlit success message.

    Parameters:
        detected_objects_summary_list (list): List of detected object indices.

    Returns:
        None
    """
    detected_objects_summary = set()
    for obj in detected_objects_summary_list:
        detected_objects_summary.add(model.names[int(obj)])
    name_summary = ", ".join(detected_objects_summary)
    st.success(f"Detected Objects: {name_summary}")

def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    """
    Display the detected objects on a video frame using the YOLOv8 model.

    Args:
    - conf (float): Confidence threshold for object detection.
    - model (YoloV8): A YOLOv8 object detection model.
    - st_frame (Streamlit object): A Streamlit object to display the detected video.
    - image (numpy array): A numpy array representing the video frame.
    - is_display_tracking (bool): A flag indicating whether to display object tracking (default=None).

    Returns:
    None
    """

    # Resize the image to a standard size
    image = cv2.resize(image, (720, int(720*(9/16))))

    # Display object tracking, if specified
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Predict the objects in the image using the YOLOv8 model
        res = model.predict(image, conf=conf)

    # # Plot the detected objects on the video frame
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )

def play_stored_video(conf, model):
    """
    Plays a stored video file. Tracks and detects objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_vid = st.sidebar.selectbox(
        "Choose a video...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))

def play_rtsp_stream(conf, model):
    """
    Plays an rtsp stream. Detects Objects in real-time using the YOLOv8 object detection model.

    Parameters:
        conf: Confidence of YOLOv8 model.
        model: An instance of the `YOLOv8` class containing the YOLOv8 model.

    Returns:
        None

    Raises:
        None
    """
    source_rtsp = st.sidebar.text_input("rtsp stream url:")
    st.sidebar.caption('Example URL: rtsp://admin:12345@192.168.1.210:554/Streaming/Channels/101')
    is_display_tracker, tracker = display_tracker_options()
    if st.sidebar.button('Detect Objects'):
        try:
            vid_cap = cv2.VideoCapture(source_rtsp)
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    # vid_cap = cv2.VideoCapture(source_rtsp)
                    # time.sleep(0.1)
                    # continue
                    break
        except Exception as e:
            vid_cap.release()
            st.sidebar.error("Error loading RTSP stream: " + str(e))