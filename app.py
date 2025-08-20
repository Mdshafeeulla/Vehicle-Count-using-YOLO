import streamlit as st
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

# Streamlit App Configuration
st.set_page_config(layout="wide", page_title="Vehicle Counting App")

st.title("Vehicle Counting and Tracking App")
st.write("This app uses YOLOv8 to detect and track vehicles in a video.")

# If we want we can use the camera also
# cap = cv2.VideoCapture(0)

# We need to take the same height and width for mask and the image or else we cant perform bitwise operation
display_width = 1280
display_height = 720

# Setting up the path for the video
uploaded_file = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])
if uploaded_file is not None:
    # Save the uploaded file to a temporary file
    with open("temp_video.mp4", "wb") as f:
        f.write(uploaded_file.getbuffer())
    cap = cv2.VideoCapture("temp_video.mp4")
else:
    cap = cv2.VideoCapture('./traffic.mp4')

mask = cv2.imread('./mask.png')
mask = cv2.resize(mask, (display_width, display_height))

# Tracking the image for counting out the vehicles
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# After the cars cross this line the will be counted
line_y = st.slider("Set the counting line's vertical position", 0, display_height, 397)
limits = [350, line_y, 1150, line_y]
total_count = []
# Store the track history
track_history = {}

# This are the classes that yolo can define by seeing
classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
           "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
           "tie", "suitcase", "frisbee", "skis", "spowboard", "sports ball", "kite", "baseball bat",
           "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife",
           "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
           "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable",
           "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
           "hair drier", "toothbrush"]

# Pre trained YOLO model here we are using the large model
model = YOLO('./yolo_weights/yolov8l.pt')

# Create a placeholder for the video
frame_placeholder = st.empty()

# Looping the video
while True:
    success, img = cap.read()
    if not success:
        st.write("Video processing finished.")
        break

    img = cv2.resize(img, (display_width, display_height))

    # The reason for using cv2.bitwise_and is to focus detection only on the region of interest, which helps prevent
    # unnecessary counting of vehicles outside the relevant area and reduces computation on irrelevant objects that
    # don't provide useful insights for this task.
    img_region = cv2.bitwise_and(img, mask)
    results = model(img_region, stream=True)

    detections = np.empty((0, 5))
    # This is used to create the bounding boxes
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1

            # This is used to show the confidence interval
            conf = math.ceil((box.conf[0] * 100)) / 100
            
            # This is used to show which object it is
            cls = int(box.cls[0])
            name = classes[cls]
            if name in ['car', 'bicycle', 'motorbike', 'bus', 'truck', 'stop sign'] and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    resultsTracker = tracker.update(detections)
    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 3)

    for result in resultsTracker:
        x1, y1, x2, y2, Id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 255, 0))
        cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=0.7, thickness=1, offset=10)

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (2, 0, 25), cv2.FILLED)

        # Store and check direction
        if Id not in track_history:
            track_history[Id] = []
        track_history[Id].append(cy)

        # Counting the number of vehicles passed
        if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
            # Check direction of movement (only count if moving down)
            if len(track_history[Id]) > 1:
                if track_history[Id][-2] < cy:
                    if total_count.count(Id) == 0:
                        total_count.append(Id)
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (255, 0, 255), 3)

    cvzone.putTextRect(img, f'count : {len(total_count)}', (50, 50),
                       scale=2, thickness=2,
                       colorT=(255, 255, 255),  # White text
                       colorR=(0, 0, 0),  # Black rectangle
                       border=3,
                       colorB=(255, 255, 255))  # Magenta border

    # Convert the image to RGB before displaying
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(img_rgb, channels="RGB", use_column_width=True)

cap.release()
