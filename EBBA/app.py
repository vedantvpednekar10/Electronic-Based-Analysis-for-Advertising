"""Streamlit dashboard for recorded video and local webcam analysis."""

import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import streamlit as st

from pipeline import FacePose, analyze


st.set_page_config(page_title="EBBA Viewer Analytics", page_icon="📊", layout="wide")
st.title("EBBA Viewer Analytics")
st.caption("Estimate billboard viewing from video using head direction.")

with st.sidebar:
    st.header("Analysis settings")
    mode = st.radio("Video source", ["Upload a video", "Local webcam"])
    yaw = st.slider("Billboard direction left / right (°)", -60, 60, 0)
    pitch = st.slider("Billboard direction up / down (°)", -45, 45, 0)
    tolerance = st.slider("Angle tolerance (°)", 5, 45, 20)
    dwell = st.slider("Minimum viewing time (seconds)", 0.0, 3.0, 0.5, 0.1)

upload = None
duration = 20
if mode == "Upload a video":
    upload = st.file_uploader("Choose a recorded video", type=["mp4", "mov", "avi", "mkv"])
else:
    st.caption("Uses the webcam attached to the computer running Streamlit.")
    duration = st.slider("Capture duration (seconds)", 5, 60, 20)

if st.button("Analyze video" if mode == "Upload a video" else "Start webcam", type="primary",
             disabled=mode == "Upload a video" and upload is None):
    st.session_state.pop("result", None)
    default = Path(__file__).parent / "models" / "shape_predictor_68_face_landmarks.dat"
    model_path = Path(os.environ.get("EBBA_LANDMARK_MODEL", default)).expanduser()
    if not model_path.is_file():
        st.error(f"Landmark model missing. Add the predictor file at {model_path}.")
    else:
        preview = st.empty()
        try:
            with st.spinner("Analyzing video…"), TemporaryDirectory() as scratch:
                model = FacePose(model_path)
                source = 0
                if upload is not None:
                    path = Path(scratch) / ("input" + Path(upload.name).suffix.lower())
                    upload.seek(0)
                    with path.open("wb") as saved:
                        shutil.copyfileobj(upload, saved)
                    source = str(path)
                samples = []
                for frame, row in analyze(source, model, yaw=yaw, pitch=pitch,
                                          tolerance=tolerance, dwell=dwell, live_seconds=duration):
                    samples.append(row)
                    preview.image(frame, channels="BGR", width="stretch")
                st.session_state.result = (frame, samples)
            preview.empty()
        except (ImportError, OSError, RuntimeError, ValueError) as error:
            preview.empty()
            st.error(f"Analysis could not finish. {error}")

if "result" in st.session_state:
    frame, samples = st.session_state.result
    report = pd.DataFrame(samples)
    last = samples[-1]
    st.subheader("Analysis results")
    detected, viewers, peak = st.columns(3)
    detected.metric("People tracked", last["tracked_people"])
    viewers.metric("Estimated viewers", last["estimated_viewers"])
    peak.metric("Peak viewers at once", int(report["active_viewers"].max()))
    st.image(frame, channels="BGR", caption="Last analyzed frame", width="stretch")
    st.line_chart(report, x="seconds", y="active_viewers", x_label="Video time (seconds)",
                  y_label="Active estimated viewers")
    st.download_button("Download timeline CSV", report.to_csv(index=False),
                       file_name="ebba-viewership.csv", mime="text/csv")
