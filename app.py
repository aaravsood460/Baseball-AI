import os
import urllib.request
import tempfile
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
st.write("Upload a baseball pitching video to track skeletal landmarks and analyze joint angles.")

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
# 3. Streamlit Interface & Processing
# -----------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload Pitching Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    # Save input video
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Temporary output video file
    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    progress_bar = st.progress(0, text="Processing pitching mechanics...")
    
    max_elbow, max_knee, max_shoulder, max_trunk = 0, 0, 0, 0
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((frame_idx / fps) * 1000)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                h, w, _ = frame.shape

                shoulder = [landmarks[12].x * w, landmarks[12].y * h]
                elbow = [landmarks[14].x * w, landmarks[14].y * h]
                wrist = [landmarks[16].x * w, landmarks[16].y * h]
                hip = [landmarks[24].x * w, landmarks[24].y * h]
                knee = [landmarks[26].x * w, landmarks[26].y * h]
                ankle = [landmarks[28].x * w, landmarks[28].y * h]

                elbow_angle = int(calculate_angle(shoulder, elbow, wrist))
                knee_angle = int(calculate_angle(hip, knee, ankle))
                shoulder_angle = int(calculate_angle(hip, shoulder, elbow))
                trunk_tilt = int(180.0 - calculate_angle(shoulder, hip, knee))

                max_elbow = max(max_elbow, elbow_angle)
                max_knee = max(max_knee, knee_angle)
                max_shoulder = max(max_shoulder, shoulder_angle)
                max_trunk = max(max_trunk, trunk_tilt)

                # Draw Overlay Skeletons
                cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), (0, 255, 0), 4)
                cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), (0, 255, 0), 4)
                cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 0, 0), 4)
                cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankle[0]), int(ankle[1])), (255, 0, 0), 4)

                for lm in [shoulder, elbow, wrist, hip, knee, ankle]:
                    cv2.circle(frame, (int(lm[0]), int(lm[1])), 7, (0, 0, 255), -1)

                # Burn angle overlays directly onto top-left of video frame
                cv2.putText(frame, f"Elbow: {elbow_angle} deg", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Knee: {knee_angle} deg", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            out.write(frame)
            frame_idx += 1
            if total_frames > 0:
                progress_bar.progress(min(int((frame_idx / total_frames) * 100), 100))

    cap.release()
    out.release()
    progress_bar.empty()

    # Layout Output View
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📺 Analyzed Pitch Video")
        st.video(output_video_path)

    with col2:
        st.subheader("📊 Peak Biomechanics Summary")
        st.caption("Key biomechanical angles detected throughout the pitch motion.")
        
        st.metric("Max Elbow Flexion Angle", f"{max_elbow}°", help="Benchmark at foot strike: 80° – 105°")
        st.metric("Max Shoulder Abduction Angle", f"{max_shoulder}°", help="Benchmark at foot strike: 85° – 100°")
        st.metric("Max Lead Knee Extension", f"{max_knee}°", help="Benchmark at release: 160° – 180°")
        st.metric("Max Trunk Forward Tilt", f"{max_trunk}°", help="Benchmark near release: 30° – 50°")

    os.remove(tfile.name)
else:
    st.info("👈 Please upload a pitching video from the sidebar to begin analysis.")