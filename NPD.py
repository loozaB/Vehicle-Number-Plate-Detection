import streamlit as st
import cv2
import tempfile
import os
from ultralytics import YOLO
import pytesseract
import numpy as np

# Load YOLOv8 models 
model = YOLO('Plate.pt')     # Number Plate Detection model
char_model = YOLO('Number.pt')   # Character Detection model (update path if needed)

# Function to detect number plate and extract numbers from a frame/image
def process_frame(frame):
    results = model.predict(frame)

    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == 'NumberPlate':
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0].item()

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Plate {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Crop the plate region
                plate_region = frame[y1:y2, x1:x2]
                if plate_region.size == 0:
                    continue

                # Optional: Resize if needed
                # resized_plate = cv2.resize(plate_region, (640, 640))

                # Run character detection on the cropped plate
                char_results = char_model.predict(plate_region, conf=0.25)[0]
                chars = []

                for cbox in char_results.boxes:
                    cx1, cy1, cx2, cy2 = map(int, cbox.xyxy[0])
                    char_cls = int(cbox.cls[0])
                    char_label = char_model.names[char_cls]

                    # Bound checking
                    ph, pw = plate_region.shape[:2]
                    cx1, cy1 = max(0, cx1), max(0, cy1)
                    cx2, cy2 = min(pw, cx2), min(ph, cy2)

                    char_img = plate_region[cy1:cy2, cx1:cx2]
                    if char_img.size == 0:
                        continue

                    chars.append((cx1, char_label))

                    # Draw character box and label
                    cv2.rectangle(plate_region, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1)
                    cv2.putText(plate_region, char_label, (cx1, cy1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # Sort and combine detected characters
                chars.sort(key=lambda tup: tup[0])
                extracted_text = ''.join([c[1] for c in chars])

                # if not extracted_text:
                #     # Optional fallback to Tesseract OCR
                #     ocr_text = pytesseract.image_to_string(plate_region, config='--psm 7')
                #     extracted_text = ocr_text.strip()

                if extracted_text:
                    # Background box for text
                    cv2.rectangle(frame, (x1, y2 + 5), (x1 + 200, y2 + 30), (255, 255, 255), -1)
                    cv2.putText(frame, f"Number: {extracted_text}", (x1, y2 + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    st.markdown(f"### 🔍 Detected Number: `{extracted_text}`")

    return frame

# Function to handle videos
def detect_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame)
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        stframe.image(processed_frame, channels="RGB")
    cap.release()

# Function to handle single image
def detect_from_image(image):
    processed_frame = process_frame(image)
    processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    st.image(processed_frame, channels="RGB")

# Streamlit UI
st.title("Number Plate & Character Detection with YOLOv8")

uploaded_file = st.file_uploader("Upload a video or image", type=["mp4", "jpg", "jpeg", "png"])

if uploaded_file:
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if file_ext == ".mp4":
        detect_from_video(tfile.name)
    elif file_ext in [".jpg", ".jpeg", ".png"]:
        image = cv2.imread(tfile.name)
        detect_from_image(image)
    else:
        st.error("Unsupported file format.")
