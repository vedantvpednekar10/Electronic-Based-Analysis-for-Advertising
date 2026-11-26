# EBBA Viewer Analytics

Analyze a recording or a local webcam session to estimate how many people face a billboard. The dashboard shows an annotated frame, cumulative viewer counts, a timeline of simultaneous viewers, and a downloadable CSV.

## Run locally

Use 64-bit Python 3.11 or 3.12. Dlib is built from source. On Windows, install Visual Studio Build Tools with the Desktop development with C++ workload and the C++ CMake tools component. Open Developer PowerShell for Visual Studio and check that `cmake --version` works before installing the Python dependencies.

```powershell
cd EBBA
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
New-Item -ItemType Directory -Path models -Force
```

Download the [Dlib landmark model archive](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2) and extract it to `models/shape_predictor_68_face_landmarks.dat`. You can also set `EBBA_LANDMARK_MODEL` to an existing predictor file.

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Measure a clip

Choose a video or the local webcam. The default settings assume the billboard and camera are aligned. Use a reference clip showing someone facing the center of the billboard to adjust the yaw and pitch offsets. Set the angle tolerance and minimum viewing time for the camera placement.

MTCNN detects faces, Dlib locates landmarks, and OpenCV estimates head direction. Short-lived IDs associate overlapping face boxes across sampled frames. The same ID contributes at most one estimated viewer per session. Long occlusions or re-entry can create new IDs, and people without a visible face can be missed. Head direction is a proxy for viewing.

## Evaluate

Create a CSV manifest with `path,looking` columns. Each row refers to a clip relative to the CSV and has a label of `1` if at least one person views the billboard and `0` otherwise. Keep evaluation subjects or source recordings separate from clips used to choose the settings.

```powershell
.\.venv\Scripts\python.exe evaluate.py held-out.csv --yaw 0 --pitch 0 --tolerance 20 --dwell 0.5
```

The script reports clip-level confusion counts, accuracy, precision, recall, and F1. Clip-level accuracy measures whether at least one view occurred. To validate viewer counts, separately compare counts against manually tracked viewers in held-out clips.
