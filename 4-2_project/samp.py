import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile

st.set_page_config(page_title="Drone Detection System", layout="wide")

# -------------------------------
# Load YOLO Models
# -------------------------------
@st.cache_resource
def load_models():
    drone_detector = YOLO("YOLO11n.pt")
    drone_classifier = YOLO("best.pt")
    return drone_detector, drone_classifier

detector, classifier = load_models()

# -------------------------------
# Helper Function
# -------------------------------
def process_frame(frame):
    results = detector(frame, conf=0.4)
    output = frame.copy()

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = frame[y1:y2, x1:x2]

            label = "Unknown"
            cls_results = classifier(crop, conf=0.25)

            if len(cls_results[0].boxes) > 0:
                cls_id = int(cls_results[0].boxes.cls[0])
                label = classifier.names[cls_id]

            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                output,
                f"Drone | {label}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    return output

# -------------------------------
# UI
# -------------------------------
st.title("🚁 Drone Detection & Type Classification")

mode = st.sidebar.selectbox(
    "Select Input Mode",
    ["Image", "Video", "Webcam"]
)

# -------------------------------
# IMAGE MODE
# -------------------------------
if mode == "Image":
    file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if file:
        img = Image.open(file).convert("RGB")
        frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        result = process_frame(frame)

        st.image(
            cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
            caption="Detection Result",
            use_container_width=True
        )

# -------------------------------
# VIDEO MODE
# -------------------------------
elif mode == "Video":
    file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov"])

    if file:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(file.read())
        tfile.close()

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = process_frame(frame)
            stframe.image(
                cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                use_container_width=True
            )

        cap.release()
        import os
        try:
            os.remove(tfile.name)
        except:
            pass

# -------------------------------
# WEBCAM MODE
# -------------------------------
elif mode == "Webcam":
    run = st.checkbox("▶ Start Webcam")

    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not accessible")
            break

        result = process_frame(frame)
        stframe.image(
            cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
            use_container_width=True
        )

    cap.release()