Object Detection and Tracking

Project Overview

This project is an AI-powered application that detects objects in images and tracks objects in videos using a pretrained YOLO model.

Objectives

- Detect objects in uploaded images.
- Track objects across video frames.
- Display detection results through a simple web interface.

Features

- Image object detection
- Video object tracking
- Bounding boxes and object labels
- Processed video preview and download

Technologies Used

- Python
- Streamlit
- Ultralytics YOLO
- OpenCV

How to Run

1. Install Python.

2. Install the required libraries:
   
   "python -m pip install -r requirements.txt"

3. Run the application:
   
   "python -m streamlit run app.py"

4. Open the local URL shown in the terminal.

Note

The application uses a pretrained YOLO model. The model may download automatically the first time it runs. Detection accuracy depends on the model and input quality.

Author

Parnika Goel