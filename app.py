import streamlit as st
import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import contextlib
import os
import gdown

st.set_page_config(page_title="CamMask AI", layout="wide")
MODEL_PATH = "new_best1.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        file_id = "1xkJNijrqTw485in8Zdd7TMBhWWwv-9Kr"
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, MODEL_PATH, quiet=False)

download_model()  # Make sure model is downloaded before loading



st.markdown("""
<style>
    .main-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    .stRadio > div {
        display: flex;
        justify-content: center;
    }
    .stButton button {
        display: block;
        margin: 0 auto;
    }
    .stImage {
        margin: 0 auto;
    }
    .alert-box {
        padding: 10px;
        border-radius: 4px;
        background-color: #e7f8ff;
        color: #31708f;
        border: 1px solid #bce8f1;
        text-align: center;
        margin: 10px auto;
        max-width: 80%;
    }
    .title {
        text-align: center;
    }
    .divider {
        width: 80%;
        margin: 10px auto;
        border-top: 1px solid #ddd;
    }
</style>
""", unsafe_allow_html=True)

@contextlib.contextmanager
def torch_load_patch():
    original_load = torch.load
    def patched_load(f, *args, **kwargs):
        kwargs["weights_only"] = False
        return original_load(f, *args, **kwargs)
    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load

@st.cache_resource
def load_model():
    with torch_load_patch():
        return YOLO("new_best1.pt")

model = load_model()

def predict(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    results = model.predict(source=image, imgsz=412, conf=0.5, iou=0.45)
    return results[0]

def visualize_detection(image, boxes):
    img_np = np.array(image)
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    elif img_np.shape[2] == 4:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2RGB)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        conf = box.conf[0]
        label = f"{model.names[cls]}: {conf:.2f}"
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 165, 0)]
        color = colors[cls] if cls < len(colors) else (0, 255, 255)

        box_width = x2 - x1
        box_height = y2 - y1
        font_scale = max(min(box_height / 60, 1.2), 0.4)
        thickness = max(int(box_height / 50), 1)

        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, thickness)
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img_np, (x1, y1 - text_height - baseline - 5), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(img_np, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return img_np, [model.names[int(box.cls[0])] for box in boxes]

with st.container():
    st.markdown('<h1 class="title">CamMask AI</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center;">Detect faces with precision</p>', unsafe_allow_html=True)
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        option = st.radio("Choose input type:", ("Upload Image", "Use Webcam Stream"))

    if option == "Upload Image":
        col1, col2, col3 = st.columns([1, 3, 1])
        with col2:
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)

            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                if st.button("🔍 Detect Faces"):
                    with st.spinner("Running detection..."):
                        result = predict(image)
                        boxes = result.boxes

                        if len(boxes) > 0:
                            img_np, class_names = visualize_detection(image, boxes)
                            st.image(img_np, caption="Detection Result", use_column_width=True)
                            st.success(f"Detected {len(boxes)} face(s).")
                            
                            unique_classes = list(set(class_names))
                            alert_message = f"Detected classes: {', '.join(unique_classes)}"
                            st.markdown(f"""
                            <div class="alert-box">
                                {alert_message}
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.warning("No faces detected.")

    elif option == "Use Webcam Stream":
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            start_stream = st.button("Start Webcam")
        
        if start_stream:
            stframe = st.empty()
            cap = cv2.VideoCapture(0)

            if not cap.isOpened():
                st.error("Cannot open webcam")
            else:
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    stop_button = st.button("Stop Streaming")
                
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Failed to grab frame")
                        break

                    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = predict(image)
                    boxes = result.boxes
                    img_np, _ = visualize_detection(image, boxes)
                    stframe.image(img_np, channels="RGB", use_column_width=True)

                    if stop_button:
                        break

                cap.release()
                st.success("Streaming stopped.")