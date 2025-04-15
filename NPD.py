import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import pytesseract

# Load YOLOv8 models 
model = YOLO('best (1).pt')     # For Number Plate Detection
char_model = YOLO('number2.pt')     # For Character Detection


# Function to detect number plate and extract numbers
def detect_numbers_and_extract(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

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

                    # Crop the plate
                    plate_region = frame[y1:y2, x1:x2]

                    # Run character detection
                    char_results = char_model.predict(plate_region)[0]

                    chars = []
                    for cbox in char_results.boxes:
                        cx1, cy1, cx2, cy2 = map(int, cbox.xyxy[0])
                        char_cls = int(cbox.cls[0])
                        char_label = char_model.names[char_cls]
                        
                        # Prevent out-of-bounds crops
                        ph, pw = plate_region.shape[:2]
                        cx1, cy1 = max(0, cx1), max(0, cy1)
                        cx2, cy2 = min(pw, cx2), min(ph, cy2)
                        
                        char_img = plate_region[cy1:cy2, cx1:cx2]
                        if char_img.size == 0:
                            continue

                        chars.append((cx1, char_label))  # store x-pos for sorting

                        # # Draw box and label
                        # cv2.rectangle(plate_region, (cx1, cy1), (cx2, cy2), (255, 0, 0), 1)
                        # cv2.putText(plate_region, char_label, (cx1, cy1 - 5),
                        #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    # Sort and combine characters
                    chars.sort(key=lambda tup: tup[0])
                    extracted_text = ''.join([c[1] for c in chars])

                    if extracted_text:
                        cv2.putText(frame, f"Number: {extracted_text}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        st.markdown(f"### 🔍 Detected Number: `{extracted_text}`")


        # Show frame in Streamlit
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame, channels="RGB")

    cap.release()

# Streamlit UI

st.title("Number Plate and Number Extraction using YOLOv8")
uploaded_video = st.file_uploader("Upload a Photo/Video", type=["mp4", "jpg"])


if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    detect_numbers_and_extract(tfile.name)
