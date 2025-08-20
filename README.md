<img width="1906" height="1059" alt="image" src="https://github.com/user-attachments/assets/6ed239e2-705b-4ad8-a925-d52d1fe3e8b9" />
# YOLOv8 Traffic Analytics Streamlit App

This is a Streamlit application that uses YOLOv8 for real-time traffic analytics.

## Features

- **Real-time video analytics:** Detects and tracks vehicles in a video stream.
- **Draggable counting bar:** A slider to position a virtual trip-line for counting vehicles.
- **Per-class filters:** Checkboxes to select which vehicle types to count.
- **Live KPI overlay:** Displays total vehicles, class-wise counts, and FPS.
- **Video Export:** Export the annotated video with a single click.

## How to Run

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

3.  **Open the app in your browser, upload a video, and see the magic!**
