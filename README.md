EBAA - Electronic Based Analysis for Advertising

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red)](https://streamlit.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green)](https://opencv.org/)

📊 Project Overview

EBAA (Electronic Based Analysis for Advertising) or AdvertisementVision is an advanced computer vision system that accurately measures billboard advertising effectiveness by tracking and analyzing viewer engagement in real-time. The system distinguishes between people who are actively viewing billboards versus those who are simply passing by.

Accuracy Achieved: 98.38%

✨ Key Features

-Real-time Person Detection - Using YOLOv4 for accurate people counting
-Face Detection & Recognition - MTCNN for robust face detection
-Engagement Analysis - Head pose estimation to determine viewing direction
-Unique Viewer Tracking - Avoid counting the same person multiple times
-Live Dashboard - Streamlit-based interface for real-time monitoring
-Video Processing - Support for pre-recorded video analysis

🛠️ Technology Stack

Computer Vision: OpenCV, MTCNN, Dlib
Object Detection: YOLOv4
Face Recognition: face_recognition, Dlib 68-point landmarks
Tracking: Kalman Filter
Head Pose: PnP (Perspective-n-Point) algorithm
UI Framework: Streamlit
Language: Python 3.8+

📁 Project Structure
```
EBAA/
├── app.py                    # Main Streamlit application
├── preprocess.py            # Video preprocessing module
├── liveFeed.py              # Live camera feed processing
├── accuracy.py              # Accuracy calculation module
├── yolov3.weights           # YOLO model weights (download separately)
├── yolov3.cfg              # YOLO configuration
├── coco.names              # COCO dataset labels
├── shape_predictor_68_face_landmarks.dat  # Dlib model
├── Unique_faces/           # Detected unique faces storage
├── output_frames/          # Processed frames output
├── Input_frames/           # Input video directory
└── videos/                 # Streamlit upload directory
```

🚀 Installation & Setup

Prerequisites
- Python 3.8 or higher
- Webcam (for live feed)
- GPU (optional, for faster processing)

Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/EBAA.git
cd EBAA
```

Step 2: Create Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Step 4: Download Model Files

Due to file size limitations, download these model files manually:

1. YOLOv3 Weights(~237 MB)
   - Download from: https://pjreddie.com/media/files/yolov3.weights
   - Place in root EBAA directory

2. Dlib Shape Predictor(~99 MB)
   - Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
   - Extract and place the .dat file in root directory

Step 5: Create Required Directories
```bash
mkdir Unique_faces output_frames Input_frames videos
```

💻 Usage

Running the Application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`

Application Features

1. Live Feed Tab
   - Real-time camera feed analysis
   - Live viewer counting
   - Engagement tracking

2. Pre-Process Tab
   - Upload and analyze video files
   - Get detailed statistics
   - View results with bounding boxes

📈 How It Works

1. Person Detection: YOLOv4 detects all people in the frame
2. Face Detection: MTCNN identifies faces from detected persons
3. Face Recognition: Assigns unique IDs to avoid duplicate counting
4. Landmark Detection: Dlib detects 68 facial landmarks
5. Head Pose Estimation: PnP algorithm calculates viewing angles
6. Engagement Analysis: Determines if person is looking at billboard
7. Tracking: Kalman filter ensures smooth tracking across frames

🎯 Performance Metrics

- Precision: 100%
- Recall: 98.38%
- F1 Score: 99.18%
- Accuracy: 98.38%

🔧 Configuration

Viewing Angle Thresholds
The system considers a person as "viewing" the billboard when:
- Yaw angle: 139° - 175°
- Pitch angle: > 1°
- Roll angle: 0° - 10°

These can be adjusted in the code based on billboard placement.

📊 Output

The system provides:
- Total number of unique faces detected
- Number of people actively viewing the billboard
- Individual face IDs with viewing status
- Processed video frames with bounding boxes
- Real-time statistics dashboard

🐛 Troubleshooting

Common Issues

1. ImportError for dlib
```bash
   pip install cmake
   pip install dlib
```

2. CUDA/GPU errors
   - System works on CPU, but GPU recommended for real-time processing
   - Install CUDA toolkit for GPU acceleration

3. Model files not found
   - Ensure all model files are in the root EBAA directory
   - Check file names match exactly

👥 Team Members

- Prathamesh Parab 
- Karan Kamath 
- Vedant Pednekar 

Guide: Mrs. Jayashri Mittal  
Institution: St. Francis Institute of Technology  
University: University of Mumbai  
Academic Year: 2023-2024

🎓 Academic Project

This project was developed as a final year B.E. Computer Engineering project and achieved:
- Successfully delivered in May 2024
- Accuracy of 98.38% in real-world testing
- Practical implementation ready for deployment


🙏 Acknowledgments

- St. Francis Institute of Technology for providing resources
- Mrs. Jayashri Mittal for guidance and mentorship
- Open-source community for amazing computer vision libraries

📧 Contact

For any queries regarding this project:
- Email: prparab@syr.edu
- Linkedin: https://www.linkedin.com/in/prathamesh-parab-3a1b32236/
- github: https://github.com/Prathu270710

---

⭐ If you find this project useful, please consider giving it a star!

🔗 Note: This is an academic project developed for educational purposes and also if you need the dataset contact at - prparab@syr.edu