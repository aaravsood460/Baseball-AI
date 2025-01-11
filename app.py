import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="AI Pitching Mechanics Tracker",
    page_icon="⚾",
    layout="wide"
)

st.title("⚾ AI Pitching Mechanics & Biomechanics Analyzer")
st.write("Upload a baseball pitching video to track skeletal landmarks and analyze joint angles in real-time.")

# --- HELPER: VECTOR ANGLE CALCULATION ---
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    return int(np.degrees(np.arccos(cosine_angle)))

# --- SIDEBAR & FILE UPLOADER ---
st.sidebar.header("⚙️ Settings & Inputs")
uploaded_file = st.sidebar.file_uploader("Choose a pitching video...", type=["mp4", "mov", "avi"])

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24),                     
    (23, 25), (25, 27), (24, 26), (26, 28)            
]

# --- MAIN PROCESSING PIPELINE ---
if uploaded_file is not None:
    # Save uploaded file temporarily so OpenCV can read it
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # Create layout columns for UI
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📹 AI Processing View")
        st_frame = st.empty() # Placeholder for video frames

    with col2:
        st.subheader("📊 Live Biomechanics Metrics")
        elbow_metric = st.empty()
        shoulder_metric = st.empty()
        knee_metric = st.empty()

    # Initialize MediaPipe Tasks Engine
import urllib.request
import os

model_path = 'pose_landmarker_full.task'
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
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
BaseOptions = python.BaseOptions

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO
)
    )

    cap = cv2.VideoCapture(video_path)

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.success("Analysis Complete!")
                break

            h, w, _ = frame.shape
            timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = landmarker.detect_for_video(mp_image, timestamp_ms)

            e_angle, s_angle, k_angle = "N/A", "N/A", "N/A"

            if results.pose_landmarks:
                for landmark_list in results.pose_landmarks:
                    points = {}
                    for idx, landmark in enumerate(landmark_list):
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        points[idx] = (cx, cy)
                        cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

                    for connection in POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        if start_idx in points and end_idx in points:
                            cv2.line(frame, points[start_idx], points[end_idx], (0, 0, 255), 2)

                    # Calculate joint angles
                    if 12 in points and 14 in points and 16 in points:
                        e_angle = calculate_angle(points[12], points[14], points[16])
                    if 24 in points and 12 in points and 14 in points:
                        s_angle = calculate_angle(points[24], points[12], points[14])
                    if 24 in points and 26 in points and 28 in points:
                        k_angle = calculate_angle(points[24], points[26], points[28])

            # Display updated video frame in Column 1
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update dashboard metrics in Column 2
            elbow_metric.metric("Right Elbow Flexion", f"{e_angle}°")
            shoulder_metric.metric("Right Shoulder Abduction", f"{s_angle}°")
            knee_metric.metric("Right Lead Knee Angle", f"{k_angle}°")

    cap.release()
    os.remove(video_path) # Clean up temp file
else:
    st.info("👈 Please upload a video file in the sidebar to start analysis.")