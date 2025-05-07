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
    """Calculates 2D interior angle (0-180 deg) between 3 joint points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

# -----------------------------------------------------------------------------
# 3. Streamlit Interface & Processing
# -----------------------------------------------------------------------------
uploaded_file = st.sidebar.file_uploader("Upload Pitching Video", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_file.read())
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".webm").name
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (orig_width, orig_height))

    progress_bar = st.progress(0, text="Processing pitching mechanics...")
    
    elbow_angles, shoulder_angles, knee_angles, trunk_tilts = [], [], [], []
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert OpenCV frame BGR -> RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((frame_idx / fps) * 1000)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]
                # Get dimensions AFTER frame adjustments
                h, w, _ = frame.shape

                # Track Right Arm / Right Leg Joints (12: R Shoulder, 14: R Elbow, 16: R Wrist)
                # (If pitcher is left-handed, use 11, 13, 15 for left side)
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

                # Store for valid aggregate analysis
                elbow_angles.append(elbow_angle)
                shoulder_angles.append(shoulder_angle)
                knee_angles.append(knee_angle)
                trunk_tilts.append(trunk_tilt)

                # Draw Overlay Skeletons accurately on body landmarks
                cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), (0, 255, 0), 3)
                cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), (0, 255, 0), 3)
                cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 0, 0), 3)
                cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankle[0]), int(ankle[1])), (255, 0, 0), 3)

                for lm in [shoulder, elbow, wrist, hip, knee, ankle]:
                    cv2.circle(frame, (int(lm[0]), int(lm[1])), 6, (0, 0, 255), -1)

                # Real-time Angle Overlays
                cv2.putText(frame, f"Elbow: {elbow_angle} deg", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Shoulder: {shoulder_angle} deg", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Knee: {knee_angle} deg", (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Trunk Tilt: {trunk_tilt} deg", (30, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            out.write(frame)
            frame_idx += 1
            if total_frames > 0:
                progress_bar.progress(min(int((frame_idx / total_frames) * 100), 100))

    cap.release()
    out.release()
    progress_bar.empty()

    # Calculate filtered peak metrics (95th percentile to eliminate outlier tracking errors)
    peak_elbow = int(np.percentile(elbow_angles, 90)) if elbow_angles else 0
    peak_shoulder = int(np.percentile(shoulder_angles, 90)) if shoulder_angles else 0
    peak_knee = int(np.percentile(knee_angles, 90)) if knee_angles else 0
    peak_trunk = int(np.percentile(trunk_tilts, 90)) if trunk_tilts else 0

    # Layout Output View
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📺 Analyzed Pitch Video")
        st.video(output_video_path)

    with col2:
        st.subheader("📊 Peak Biomechanics Summary")
        st.caption("Realistic peak mechanical values across pitch execution (filtered for accuracy).")
        
        st.metric("Max Elbow Flexion Angle", f"{peak_elbow}°", help="Benchmark at foot strike: 80° – 105°")
        st.metric("Max Shoulder Abduction Angle", f"{peak_shoulder}°", help="Benchmark at foot strike: 85° – 100°")
        st.metric("Max Lead Knee Extension", f"{peak_knee}°", help="Benchmark at release: 160° – 180°")
        st.metric("Max Trunk Forward Tilt", f"{peak_trunk}°", help="Benchmark near release: 30° – 50°")

    os.remove(tfile.name)
else:
    st.info("👈 Please upload a pitching video from the sidebar to begin analysis.")