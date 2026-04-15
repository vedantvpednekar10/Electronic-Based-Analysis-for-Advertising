# 📊 EBAA - Electronic Based Analysis for Advertising

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green)

---

## 🚀 Project Overview

**EBAA (Electronic Based Analysis for Advertising)**, also known as *AdvertisementVision*, is an advanced computer vision system designed to measure billboard advertising effectiveness by analyzing real-time viewer engagement.

The system distinguishes between:

* Users actively viewing advertisements
* Users simply passing by

It uses deep learning, face detection, and tracking algorithms to compute engagement metrics and optimize advertisement placement strategies.

**🎯 Accuracy Achieved: 98.38%**

---

## ✨ Key Features

* 👁️ Real-time Person Detection using YOLOv4
* 🧠 Face Detection & Recognition using MTCNN + Dlib
* 📊 Engagement Analysis using Head Pose Estimation
* 🔁 Unique Viewer Tracking (prevents duplicate counting)
* 📡 Live Streamlit Dashboard for monitoring
* 🎥 Video processing for recorded datasets
* 📈 Analytics for advertisement effectiveness

---

## 🛠️ Technology Stack

* **Computer Vision:** OpenCV, MTCNN, Dlib
* **Object Detection:** YOLOv4
* **Face Recognition:** face_recognition, Dlib 68-point landmarks
* **Tracking:** Kalman Filter
* **Head Pose Estimation:** Perspective-n-Point (PnP) Algorithm
* **UI Framework:** Streamlit
* **Language:** Python 3.8+

---

## 📁 Project Structure

```
EBAA/
├── app.py                          # Streamlit main application
├── preprocess.py                   # Video preprocessing module
├── liveFeed.py                     # Real-time camera processing
├── accuracy.py                     # Performance evaluation
├── yolov3.weights                  # YOLO model weights
├── yolov3.cfg                      # YOLO configuration
├── coco.names                      # COCO dataset labels
├── shape_predictor_68_face_landmarks.dat
├── Unique_faces/                   # Unique face storage
├── output_frames/                  # Processed outputs
├── Input_frames/                   # Input video dataset
└── videos/                         # Uploaded videos
```

---

## ⚙️ Installation & Setup

### 🔹 Prerequisites

* Python 3.8+
* Webcam (for live mode)
* GPU (optional but recommended)

---

### 🔹 Step 1: Clone Repository

```bash
git clone https://github.com/vedantvpednekar10/Electronic-Based-Analysis-for-Advertising.git
cd Electronic-Based-Analysis-for-Advertising
```

---

### 🔹 Step 2: Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

---

### 🔹 Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 🔹 Step 4: Download Model Files

Due to size limitations, download manually:

* YOLOv3 Weights (~237 MB)
  https://pjreddie.com/media/files/yolov3.weights

* Dlib Shape Predictor (~99 MB)
  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

Place both files in the root directory.

---

### 🔹 Step 5: Create Required Folders

```bash
mkdir Unique_faces output_frames Input_frames videos
```

---

## 💻 Usage

### ▶️ Run Application

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

### 📌 Application Modules

**1. Live Feed Module**

* Real-time camera analysis
* Live viewer detection
* Engagement tracking

**2. Preprocessing Module**

* Upload video files
* Run batch analysis
* Generate metrics & bounding boxes

---

## 📈 How It Works

1. **Person Detection** → YOLOv4 detects humans in frame
2. **Face Detection** → MTCNN isolates faces
3. **Face Recognition** → Assigns unique IDs
4. **Landmark Detection** → Dlib extracts facial landmarks
5. **Head Pose Estimation** → Computes viewing direction
6. **Engagement Analysis** → Determines attention toward billboard
7. **Tracking** → Kalman filter maintains continuity

---

## 🎯 Performance Metrics

* Precision: **100%**
* Recall: **98.38%**
* F1 Score: **99.18%**
* Accuracy: **98.38%**

---

## 🔧 Configuration

### Viewing Angle Thresholds

A user is considered *“viewing the billboard”* when:

* Yaw: 139° – 175°
* Pitch: > 1°
* Roll: 0° – 10°

These values can be adjusted depending on billboard position and camera setup.

---

## 📊 Output

The system provides:

* Total unique viewers detected
* Active viewers at any moment
* Individual face tracking IDs
* Bounding box visualization
* Real-time analytics dashboard

---

## 🐛 Troubleshooting

### ❌ dlib installation error

```bash
pip install cmake
pip install dlib
```

### ❌ GPU/CUDA issues

* Works on CPU by default
* GPU recommended for real-time performance

### ❌ Model file not found

* Ensure all model files are in root directory
* Verify filenames exactly match

---

## 👥 Team Members

* Vedant Pednekar
* Prathamesh Parab
* Karan Kamath


---

## 🙏 Guide & Institution

**Guide:** Mrs. Jayashri Mittal
**Institution:** St. Francis Institute of Technology
**University:** University of Mumbai
**Academic Year:** 2023–2024

---

## 🧑‍💻 My Contributions

* Took camera recordings of test and training subjects
* Researched and implemented YOLOv4 for detection
* Worked on tracking system using Kalman Filter
* Optimized head pose estimation for angle accuracy
* Improved system performance and evaluation metrics

---

## 📧 Contact

**Vedant Pednekar**
Graduate Student | Software Developer | AI Enthusiast
Email: [vedantvpednekar10@gmail.com](mailto:vedantvpednekar10@gmail.com)
University Email: [vepednek@syr.edu](mailto:vepednek@syr.edu)
LinkedIn: https://www.linkedin.com/in/vedantpednekar/
GitHub: https://github.com/vedantvpednekar10

---

## 🎓 Academic Project

This project was developed as a final year B.E. Computer Engineering project.

* Completed: May 2024
* Accuracy: 98.38%
* Fully functional real-world prototype

---

## ⭐ Acknowledgements

* St. Francis Institute of Technology
* Project Guide: Mrs. Jayashri Mittal
* Open-source computer vision community

---

⭐ If you like this project, please consider giving it a star on GitHub!
