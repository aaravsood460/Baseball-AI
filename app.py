import os
import urllib.request
import tempfile
import time
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -----------------------------------------------------------------------------
# Streamlit Page Setup
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Pitching Mechanics Analyzer",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ AI Pitching Mechanics & Biomechanics Analyzer")
st.write("Upload a baseball pitching video to track skeletal landmarks and analyze joint angles in real-time.")

# -----------------------------------------------------------------------------
# 1. Automatic Model Setup
# -----------------------------------------------------------------------------
model_path = "pose_landmarker_full.task"

if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, model_path)

PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
BaseOptions = python.BaseOptions

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO
)

# -----------------------------------------------------------------------------
# 2. Helper Functions
# -----------------------------------------------------------------------------
def calculate_angle(a, b, c):
    """Calculates 2D angle between 3 points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360.0 - angle if angle > 180.0 else angle

# -----------------------------------------------------------------------------
# 3. Streamlit Interface
# -----------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload Pitching Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📺 AI Processing View")
        st_frame = st.empty()

    with col2:
        st.subheader("📊 Live Biomechanics Metrics")
        st.caption("Benchmark ranges reflect standard elite pitching mechanics.")
        
        elbow_metric = st.empty()
        shoulder_metric = st.empty()
        knee_metric = st.empty()
        hip_metric = st.empty()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    # Read all frames into memory first
    raw_frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    processed_frames = []
    frame_metrics = []

    progress_bar = st.progress(0, text="Analyzing pitching mechanics frames...")

    # Run AI pose estimation on frames
    with PoseLandmarker.create_from_options(options) as landmarker:
        total_frames = len(raw_frames)
        for idx, frame in enumerate(raw_frames):
            progress_bar.progress(int((idx + 1) / total_frames * 100), text=f"Analyzing frame {idx+1}/{total_frames}...")
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((idx / fps) * 1000)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            metrics = {"elbow": 0, "shoulder": 0, "knee": 0, "trunk": 0}

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                h, w, _ = frame.shape

                shoulder = [landmarks[12].x * w, landmarks[12].y * h]
                elbow = [landmarks[14].x * w, landmarks[14].y * h]
                wrist = [landmarks[16].x * w, landmarks[16].y * h]
                hip = [landmarks[24].x * w, landmarks[24].y * h]
                knee = [landmarks[26].x * w, landmarks[26].y * h]
                ankle = [landmarks[28].x * w, landmarks[28].y * h]

                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                knee_angle = calculate_angle(hip, knee, ankle)
                shoulder_angle = calculate_angle(hip, shoulder, elbow)
                trunk_tilt = 180.0 - calculate_angle(shoulder, hip, knee)

                metrics = {
                    "elbow": int(elbow_angle),
                    "shoulder": int(shoulder_angle),
                    "knee": int(knee_angle),
                    "trunk": int(trunk_tilt)
                }

                # Draw Overlay Skeletons directly onto frame
                cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), (0, 255, 0), 4)
                cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), (0, 255, 0), 4)
                cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 0, 0), 4)
                cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankle[0]), int(ankle[1])), (255, 0, 0), 4)

                for lm in [shoulder, elbow, wrist, hip, knee, ankle]:
                    cv2.circle(frame, (int(lm[0]), int(lm[1])), 7, (0, 0, 255), -1)

            processed_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_metrics.append(metrics)

    progress_bar.empty()

    # Smooth Playback Loop
    frame_delay = 1.0 / fps
    for frame, m in zip(processed_frames, frame_metrics):
        st_frame.image(frame, channels="RGB", use_container_width=True)
        
        elbow_metric.metric("Elbow Flexion Angle", f"{m['elbow']}°", help="Benchmark: 80° – 105°")
        shoulder_metric.metric("Shoulder Abduction Angle", f"{m['shoulder']}°", help="Benchmark: 85° – 100°")
        knee_metric.metric("Lead Knee Extension", f"{m['knee']}°", help="Benchmark: 160° – 180°")
        hip_metric.metric("Trunk Forward Tilt", f"{m['trunk']}°", help="Benchmark: 30° – 50°")

        time.sleep(frame_delay)

    os.remove(tfile.name)
else:
    st.info("👈 Please upload a pitching video from the sidebar to begin analysis.")