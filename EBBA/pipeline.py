"""Face detection, head pose estimation, and per-video viewer counts."""

from dataclasses import dataclass
from math import atan2, degrees, hypot

import cv2
import numpy as np


LANDMARKS = (30, 8, 36, 45, 48, 54)
# Face model in OpenCV camera coordinates for a frontal pose.
FACE_MODEL = np.array(
    [(0, 0, 0), (0, 330, 65), (-225, -170, 135),
     (225, -170, 135), (-150, 150, 125), (150, 150, 125)],
    dtype=np.float64,
)


@dataclass
class Track:
    box: tuple[int, int, int, int]
    last_seen: float
    facing_since: float | None = None


def overlap(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1])
    return width * height / (union - width * height)


class ViewerCounter:
    def __init__(self, dwell_seconds=0.5, max_gap=1.0):
        self.dwell_seconds = dwell_seconds
        self.max_gap = max_gap
        self.tracks = {}
        self.total = 0
        self.viewers = set()

    def update(self, observations, timestamp):
        self.tracks = {
            key: track for key, track in self.tracks.items()
            if timestamp - track.last_seen <= self.max_gap
        }
        candidates = sorted(
            ((overlap(box, track.box), index, key)
             for index, (box, _) in enumerate(observations)
             for key, track in self.tracks.items()),
            reverse=True,
        )
        assigned = {}
        used = set()
        for score, index, key in candidates:
            if score >= 0.2 and index not in assigned and key not in used:
                assigned[index] = key
                used.add(key)

        for key, track in self.tracks.items():
            if key not in used:
                track.facing_since = None

        current = []
        for index, (box, facing) in enumerate(observations):
            key = assigned.get(index)
            if key is None:
                self.total += 1
                key = self.total
                self.tracks[key] = Track(box, timestamp)
            track = self.tracks[key]
            track.box, track.last_seen = box, timestamp
            if not facing:
                track.facing_since = None
            elif track.facing_since is None:
                track.facing_since = timestamp
            engaged = (
                track.facing_since is not None
                and timestamp - track.facing_since >= self.dwell_seconds
            )
            if engaged:
                self.viewers.add(key)
            current.append((key, box, facing, engaged))
        return current


class FacePose:
    def __init__(self, model_path):
        from mtcnn import MTCNN
        import dlib

        self.detector = MTCNN()
        self.landmarks = dlib.shape_predictor(str(model_path))
        self.rectangle = dlib.rectangle

    def observations(self, frame, yaw_center, pitch_center, tolerance):
        height, width = frame.shape[:2]
        focal = max(width, height)
        camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]])
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        for face in self.detector.detect_faces(rgb):
            if face["confidence"] < 0.9:
                continue
            x, y, w, h = face["box"]
            left, top, right, bottom = max(0, x), max(0, y), min(width, x + w), min(height, y + h)
            if right - left < 20 or bottom - top < 20:
                continue
            shape = self.landmarks(rgb, self.rectangle(left, top, right - 1, bottom - 1))
            points = np.array([(shape.part(i).x, shape.part(i).y) for i in LANDMARKS], dtype=np.float64)
            solved, rotation, translation = cv2.solvePnP(FACE_MODEL, points, camera, None)
            facing = False
            if solved and translation[2, 0] > 0:
                # The outward face direction is the negative Z axis of the face model.
                horizontal, vertical, depth = cv2.Rodrigues(rotation)[0][:, 2]
                yaw = degrees(atan2(-horizontal, depth))
                pitch = degrees(atan2(-vertical, hypot(horizontal, depth)))
                facing = abs(yaw - yaw_center) <= tolerance and abs(pitch - pitch_center) <= tolerance
            yield (left, top, right, bottom), facing


def analyze(source, model, *, yaw=0, pitch=0, tolerance=20, dwell=0.5, live_seconds=20):
    """Yield annotated frames and counts at up to five samples per second."""
    from math import isfinite
    from time import monotonic

    capture = cv2.VideoCapture(source)
    counter = ViewerCounter(dwell)
    live = isinstance(source, int)
    start = monotonic()
    frame_index = -1
    previous = -1.0
    try:
        if not capture.isOpened():
            raise ValueError("Could not open the webcam or video file.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not isfinite(fps) or fps <= 0:
            fps = 30
        while not live or monotonic() - start < live_seconds:
            ok, frame = capture.read()
            if not ok:
                break
            frame_index += 1
            timestamp = monotonic() - start if live else frame_index / fps
            if timestamp - previous < 0.2 - 1e-9:
                continue
            previous = timestamp
            width = frame.shape[1]
            if width > 960:
                frame = cv2.resize(frame, (960, max(1, round(frame.shape[0] * 960 / width))))
            observations = list(model.observations(frame, yaw, pitch, tolerance))
            tracks = counter.update(observations, timestamp)
            for key, (left, top, right, bottom), facing, engaged in tracks:
                color = (135, 220, 45) if engaged else (235, 170, 55)
                status = "Viewing" if engaged else "Facing" if facing else "Detected"
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, f"#{key} {status}", (left, max(18, top - 7)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            yield frame, {
                "seconds": round(timestamp, 3),
                "active_viewers": sum(track[3] for track in tracks),
                "tracked_people": counter.total,
                "estimated_viewers": len(counter.viewers),
            }
        if frame_index < 0:
            raise ValueError("No video frames could be read.")
    finally:
        capture.release()
