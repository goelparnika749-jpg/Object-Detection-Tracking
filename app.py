import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from ultralytics import YOLO


# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
    page_title="AI Object Detection and Tracking",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 AI Object Detection and Tracking")

st.write(
    "Detect objects in images and track objects in videos "
    "using a pretrained YOLO model."
)


# ---------------- LOAD YOLO MODEL ----------------

@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")


try:
    model = load_model()
except Exception as e:
    st.error("YOLO model could not be loaded.")
    st.error(str(e))
    st.stop()


# ---------------- MODE SELECTION ----------------

mode = st.radio(
    "Select an option:",
    ["📷 Image Detection", "🎥 Video Tracking"],
    horizontal=True
)


# =================================================
#              IMAGE DETECTION
# =================================================

if mode == "📷 Image Detection":

    st.header("📷 Image Object Detection")

    uploaded_image = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:

        # Read uploaded image
        image_bytes = uploaded_image.getvalue()

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error("Image could not be read.")

        else:

            st.subheader("Original Image")

            st.image(
                cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

            if st.button(
                "🔍 Detect Objects",
                type="primary"
            ):

                with st.spinner(
                    "Detecting objects..."
                ):

                    results = model.predict(
                        source=image,
                        conf=0.45,
                        imgsz=640,
                        verbose=False
                    )

                result = results[0]

                # Draw bounding boxes
                annotated_image = result.plot()

                annotated_image = cv2.cvtColor(
                    annotated_image,
                    cv2.COLOR_BGR2RGB
                )

                st.subheader("Detection Result")

                st.image(
                    annotated_image,
                    use_container_width=True
                )

                # Number of objects
                boxes = result.boxes

                st.success(
                    f"Detection complete! "
                    f"{len(boxes)} object(s) detected."
                )

                # Show detected object names
                if len(boxes) > 0:

                    st.subheader(
                        "Detected Objects"
                    )

                    names = result.names

                    detected_objects = []

                    for box in boxes:

                        class_id = int(
                            box.cls.item()
                        )

                        confidence = float(
                            box.conf.item()
                        )

                        object_name = names[
                            class_id
                        ]

                        detected_objects.append(
                            (
                                object_name,
                                confidence
                            )
                        )

                    for object_name, confidence in detected_objects:

                        st.write(
                            f"✅ **{object_name}** "
                            f"— {confidence * 100:.1f}% confidence"
                        )

                else:

                    st.warning(
                        "No objects detected."
                    )


# =================================================
#              VIDEO TRACKING
# =================================================

else:

    st.header("🎥 Video Object Tracking")

    uploaded_video = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        if st.button(
            "🎯 Track Objects",
            type="primary"
        ):

            input_path = None
            output_path = None

            try:

                # Save uploaded video
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp4"
                ) as input_file:

                    input_file.write(
                        uploaded_video.getvalue()
                    )

                    input_path = input_file.name

                # Open video
                cap = cv2.VideoCapture(
                    input_path
                )

                if not cap.isOpened():

                    st.error(
                        "Video could not be opened."
                    )

                else:

                    fps = cap.get(
                        cv2.CAP_PROP_FPS
                    )

                    if fps <= 0:
                        fps = 20

                    width = int(
                        cap.get(
                            cv2.CAP_PROP_FRAME_WIDTH
                        )
                    )

                    height = int(
                        cap.get(
                            cv2.CAP_PROP_FRAME_HEIGHT
                        )
                    )

                    total_frames = int(
                        cap.get(
                            cv2.CAP_PROP_FRAME_COUNT
                        )
                    )

                    # Output video
                    output_path = tempfile.mktemp(
                        suffix=".mp4"
                    )

                    writer = cv2.VideoWriter(
                        output_path,
                        cv2.VideoWriter_fourcc(
                            *"mp4v"
                        ),
                        fps,
                        (width, height)
                    )

                    progress = st.progress(0)

                    frame_number = 0

                    with st.spinner(
                        "Tracking objects..."
                    ):

                        while True:

                            success, frame = cap.read()

                            if not success:
                                break

                            results = model.track(
                                source=frame,
                                persist=True,
                                tracker="bytetrack.yaml",
                                conf=0.45,
                                imgsz=640,
                                verbose=False
                            )

                            result = results[0]

                            annotated_frame = result.plot()

                            writer.write(
                                annotated_frame
                            )

                            frame_number += 1

                            if total_frames > 0:

                                progress.progress(
                                    min(
                                        frame_number /
                                        total_frames,
                                        1.0
                                    )
                                )

                    cap.release()
                    writer.release()

                    if frame_number == 0:

                        st.error(
                            "No frames were processed."
                        )

                    else:

                        st.success(
                            f"Tracking completed! "
                            f"{frame_number} frames processed."
                        )

                        st.subheader(
                            "Tracked Video"
                        )

                        st.video(
                            output_path
                        )

                        with open(
                            output_path,
                            "rb"
                        ) as video_file:

                            video_data = (
                                video_file.read()
                            )

                        st.download_button(
                            label="⬇️ Download Tracked Video",
                            data=video_data,
                            file_name="tracked_video.mp4",
                            mime="video/mp4"
                        )

            except Exception as e:

                st.error(
                    f"Video processing failed: {e}"
                )

            finally:

                if input_path and os.path.exists(
                    input_path
                ):

                    os.remove(input_path)


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AI Object Detection & Tracking | "
    "Python • YOLO • OpenCV • Streamlit"
)