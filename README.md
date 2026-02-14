# Real-Time Face Mask Detection 😷

## 📌 Project Overview

Real-Time Face Mask Detection is a computer vision project that detects whether a person is wearing a face mask using live video input.
The system uses **YOLOv8** for object detection and **OpenCV** for real-time webcam processing.

The goal of this project is to demonstrate practical deployment of deep learning for real-time monitoring and safety applications.

---

## 🚀 Features

* Real-time face mask detection using webcam input
* Mask vs No-Mask classification
* YOLOv8-based object detection pipeline
* Live video frame processing with OpenCV
* Fast inference suitable for real-time usage

---

## 🧠 Tech Stack

* **Programming Language:** Python
* **Deep Learning:** YOLOv8
* **Computer Vision:** OpenCV
* **Libraries:** NumPy, Ultralytics

---

## 📂 Project Structure

```
Real-Time-Face-Mask-Detector
│
├── app.py                # Main application file
├── model.ipynb           # Model training / experimentation
├── requirements.txt      # Dependencies
├── README.md
└── .gitignore
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Real-Time-Face-Mask-Detector.git
cd Real-Time-Face-Mask-Detector
```

### 2️⃣ Create virtual environment (recommended)

```bash
python -m venv myenv
```

Activate environment:

Windows:

```bash
myenv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

```bash
python app.py
```

The webcam will open and start detecting face masks in real-time.

---

## 📊 Evaluation Metrics

The model performance is evaluated using:

* Precision
* Recall
* F1-Score
* IoU (Intersection over Union)
* mAP (Mean Average Precision)

---

## 🎯 Use Cases

* Public safety monitoring
* Smart surveillance systems
* Healthcare and workplace safety
* AI-based monitoring applications

---

## 📌 Future Improvements

* Improve detection accuracy in low-light environments
* Add multi-face tracking
* Deploy as a web-based application
* Optimize model for edge devices

---

## 👨‍💻 Author

**Shivakoti Raj Harsha**
B.Tech CSE — IIIT Jabalpur
