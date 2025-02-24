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

# Download the model dynamically if missing or corrupted on Streamlit Cloud
if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
    url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    urllib.request.urlretrieve(url, model_path)

# Initialize MediaPipe Tasks Engine Options
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
BaseOptions = python.BaseOptions

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO
)

# -----------------------------------------------------------------------------
# 2. Biomechanics Helper Functions
# -----------------------------------------------------------------------------
def calculate_angle(a, b, c):
    """Calculates the 2D angle (in degrees) between three landmark points."""
    a = np.array(a)  # First joint
    b = np.array(b)  # Middle joint (vertex)
    c = np.array(c)  # End joint
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

# -----------------------------------------------------------------------------
# 3. Streamlit UI Layout & Video Upload
# -----------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload Pitching Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save uploaded file to a temporary file for OpenCV reading
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📺 AI Processing View")
        st_frame = st.empty()

    with col2:
        st.subheader("📊 Live Biomechanics Metrics")
        st.caption("Benchmark ranges reflect standard elite pitching mechanics at key motion phases.")
        
        elbow_metric = st.empty()
        shoulder_metric = st.empty()
        knee_metric = st.empty()
        hip_metric = st.empty()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    frame_timestamp_ms = 0

    # Execute MediaPipe Pose Engine Context Manager
    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert OpenCV frame BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            # Process frame with MediaPipe
            frame_timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            pose_landmarker_result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

            # Draw Landmarks & Calculate Angles
            if pose_landmarker_result.pose_landmarks:
                landmarks = pose_landmarker_result.pose_landmarks[0]
                h, w, _ = frame.shape

                # Extract Key Landmarks
                shoulder = [landmarks[12].x * w, landmarks[12].y * h]
                elbow = [landmarks[14].x * w, landmarks[14].y * h]
                wrist = [landmarks[16].x * w, landmarks[16].y * h]
                hip = [landmarks[24].x * w, landmarks[24].y * h]
                knee = [landmarks[26].x * w, landmarks[26].y * h]
                ankle = [landmarks[28].x * w, landmarks[28].y * h]

                # Compute Joint Angles
                elbow_angle = calculate_angle(shoulder, elbow, wrist)
                knee_angle = calculate_angle(hip, knee, ankle)
                shoulder_angle = calculate_angle(hip, shoulder, elbow)
                trunk_tilt = 180.0 - calculate_angle(shoulder, hip, knee)

                # Draw Overlay Skeletons
                cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), (0, 255, 0), 3)
                cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), (0, 255, 0), 3)
                cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 0, 0), 3)
                cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankle[0]), int(ankle[1])), (255, 0, 0), 3)

                for lm in [shoulder, elbow, wrist, hip, knee, ankle]:
                    cv2.circle(frame, (int(lm[0]), int(lm[1])), 6, (0, 0, 255), -1)

                # Update Streamlit Metrics Panel with Benchmark Hints
                elbow_metric.metric(
                    label="Elbow Flexion Angle", 
                    value=f"{int(elbow_angle)}°", 
                    help="Optimal benchmark at foot strike: 80° – 105°"
                )
                shoulder_metric.metric(
                    label="Shoulder Abduction Angle", 
                    value=f"{int(shoulder_angle)}°", 
                    help="Optimal benchmark at foot strike: 85° – 100°"
                )
                knee_metric.metric(
                    label="Lead Knee Extension", 
                    value=f"{int(knee_angle)}°", 
                    help="Optimal benchmark at ball release: 160° – 180°"
                )
                hip_metric.metric(
                    label="Trunk Forward Tilt", 
                    value=f"{int(trunk_tilt)}°", 
                    help="Optimal benchmark near release: 30° – 50°"
                )

            # Render updated frame back to Streamlit
            st_frame.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # Frame pacing to ensure smooth sequential playback
            time.sleep(1.0 / fps)

    cap.release()
    os.remove(tfile.name)
else:
    st.info("👈 Please upload a pitching video from the sidebar to begin analysis.")