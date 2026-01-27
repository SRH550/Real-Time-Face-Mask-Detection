# CamMask AI 🎭

**CamMask AI** is a lightweight Streamlit-based app that uses YOLOv8 to detect faces from uploaded images or a live webcam stream. It provides an intuitive interface and clear visual overlays for easy interpretation of detection results.

---

## 🔧 Features

- ✅ Upload an image and detect faces
- ✅ Live webcam face detection
- ✅ Bounding boxes with confidence scores
- ✅ Modern and clean UI with centered layout
- ✅ Efficient and lightweight YOLOv8 backend

---

## 🧠 Model Information

This app uses a custom YOLOv8 model trained for face detection.  
The model file `new_best1.pt` will be **automatically downloaded** from Google Drive when the app starts.

> ✅ No manual action needed — the model downloads on first run from this [Google Drive link](https://drive.google.com/file/d/1xkJNijrqTw485in8Zdd7TMBhWWwv-9Kr/view?usp=sharing).

---

## 📦 Installation

```bash
git clone https://github.com/Rohitpulagam/cammask_ai.git
cd cammask_ai

# Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

---

## How to run

streamlit run app.py

---

## 📝 License

This project is licensed under the MIT License.
