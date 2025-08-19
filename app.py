import streamlit as st
from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import numpy as np
import os
import requests
import time

# --- Helper Functions ---
def download_file(url, file_path):
    """Downloads a file from a URL to a local path."""
    if not os.path.exists(file_path):
        st.info(f"Downloading {os.path.basename(file_path)}...")
        if not os.path.exists(os.path.dirname(file_path)):
             os.makedirs(os.path.dirname(file_path), exist_ok=True)
        response = requests.get(url, stream=True)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        st.success(f"Downloaded {os.path.basename(file_path)}")

def export_video_file(video_path, model, line_y_position, selected_classes, all_classes):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_video_path = "annotated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    limits = [0, line_y_position, frame_width, line_y_position]
    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    total_count = []
    class_counts = {cls: 0 for cls in selected_classes}
    tracked_classes = {}

    progress_bar = st.progress(0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(total_frames):
        success, img = cap.read()
        if not success:
            break

        results = model(img, stream=True, verbose=False)
        detections = np.empty((0, 5))
        detection_classes = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                name = all_classes[cls]
                if name in selected_classes and conf > 0.3:
                    currentArray = np.array([x1, y1, x2, y2, conf])
                    detections = np.vstack((detections, currentArray))
                    detection_classes.append(name)

        resultsTracker = tracker.update(detections)
        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

        for result in resultsTracker:
            x1, y1, x2, y2, Id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
            cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=1, thickness=2, offset=10)
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            if Id not in tracked_classes:
                min_dist = float('inf')
                assigned_class = None
                for i_det, det in enumerate(detections):
                    det_x1, det_y1, det_x2, det_y2, _ = det
                    det_cx, det_cy = (det_x1 + det_x2) // 2, (det_y1 + det_y2) // 2
                    dist = math.sqrt((cx - det_cx)**2 + (cy - det_cy)**2)
                    if dist < 30:
                        min_dist = dist
                        assigned_class = detection_classes[i_det]
                if assigned_class:
                    tracked_classes[Id] = assigned_class
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if total_count.count(Id) == 0:
                    total_count.append(Id)
                    if Id in tracked_classes:
                        cls_name = tracked_classes[Id]
                        if cls_name in class_counts:
                            class_counts[cls_name] += 1
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)

        cvzone.putTextRect(img, f'Total Count: {len(total_count)}', (50, 50), scale=2, thickness=3)

        out.write(img)
        progress_bar.progress((i + 1) / total_frames)

    out.release()
    cap.release()
    return output_video_path, class_counts, len(total_count)

# --- Main Application ---
st.set_page_config(page_title="YOLOv8 Traffic Analytics", layout="wide")
st.title("🚦 No-code Traffic-Analytics Tool")

with st.sidebar:
    st.header("Configuration")
    yolo_weights_path = "yolov8l.pt"
    if not os.path.exists(yolo_weights_path):
        yolo_weights_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt"
        download_file(yolo_weights_url, yolo_weights_path)
    model = YOLO(yolo_weights_path)
    all_classes = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
    default_vehicle_classes = ["bicycle", "car", "motorbike", "bus", "truck"]
    selected_classes = st.multiselect("Choose vehicle types to count", all_classes, default=default_vehicle_classes)
    line_y_position = st.slider("Reposition the counting bar", 0, 1080, 350)
    uploaded_file = st.file_uploader("Upload a 1080p video", type=["mp4", "mov", "avi"])
    export_button = st.button("Export annotated video")

if uploaded_file is not None:
    video_path = "temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if export_button:
        st.info("Exporting video... This may take a while.")
        output_path, class_counts, total_count = export_video_file(video_path, model, line_y_position, selected_classes, all_classes)
        st.success("Video export complete!")
        st.subheader("Export Summary")
        st.metric("Total Vehicles Counted", total_count)
        st.write("Class-wise Counts:")
        st.json(class_counts)
        with open(output_path, "rb") as file:
            st.download_button(label="Download Annotated Video", data=file, file_name="annotated_video.mp4", mime="video/mp4")
        if os.path.exists(output_path):
            os.remove(output_path)
    else:
        cap = cv2.VideoCapture(video_path)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        limits = [0, line_y_position, frame_width, line_y_position]
        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
        total_count = []
        class_counts = {cls: 0 for cls in selected_classes}
        tracked_classes = {}
        stframe = st.empty()
        kpi_cols = st.columns(3)
        total_count_kpi = kpi_cols[0].empty()
        class_kpi_placeholder = kpi_cols[1].empty()
        fps_kpi = kpi_cols[2].empty()
        start_time = time.time()
        frame_count = 0
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                break
            frame_count += 1
            results = model(img, stream=True, verbose=False)
            detections = np.empty((0, 5))
            detection_classes = []
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    conf = math.ceil((box.conf[0] * 100)) / 100
                    cls = int(box.cls[0])
                    name = all_classes[cls]
                    if name in selected_classes and conf > 0.3:
                        currentArray = np.array([x1, y1, x2, y2, conf])
                        detections = np.vstack((detections, currentArray))
                        detection_classes.append(name)
            resultsTracker = tracker.update(detections)
            cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)
            for result in resultsTracker:
                x1, y1, x2, y2, Id = result
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))
                cvzone.putTextRect(img, f'{int(Id)}', (max(0, x1), max(35, y1)), scale=1, thickness=2, offset=10)
                cx, cy = x1 + w // 2, y1 + h // 2
                cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                if Id not in tracked_classes:
                    min_dist = float('inf')
                    assigned_class = None
                    for i, det in enumerate(detections):
                        det_x1, det_y1, det_x2, det_y2, _ = det
                        det_cx, det_cy = (det_x1 + det_x2) // 2, (det_y1 + det_y2) // 2
                        dist = math.sqrt((cx - det_cx)**2 + (cy - det_cy)**2)
                        if dist < 30:
                            min_dist = dist
                            assigned_class = detection_classes[i]
                    if assigned_class:
                        tracked_classes[Id] = assigned_class
                if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                    if total_count.count(Id) == 0:
                        total_count.append(Id)
                        if Id in tracked_classes:
                            cls_name = tracked_classes[Id]
                            if cls_name in class_counts:
                                class_counts[cls_name] += 1
                        cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5)
            end_time = time.time()
            if (end_time - start_time) > 0:
                fps = frame_count / (end_time - start_time)
                fps_kpi.metric("FPS", f"{fps:.2f}")
            total_count_kpi.metric("Total Vehicles", len(total_count))
            class_kpi_text = "Class-wise Counts:\n"
            for cls, count in class_counts.items():
                class_kpi_text += f"- {cls}: {count}\n"
            class_kpi_placeholder.text(class_kpi_text)
            stframe.image(img, channels="BGR", use_column_width=True)
        cap.release()
    if os.path.exists(video_path):
        os.remove(video_path)
else:
    st.info("Upload a video to start.")
