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
st.write("Upload one or multiple pitching videos to calculate individual pitch peaks and establish the pitcher's personal biomechanical range.")

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

def process_single_video(file_bytes, file_name):
    """Processes a single video clip and returns path to output video and peak angles dict."""
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(file_bytes)
    tfile.close()

    cap = cv2.VideoCapture(tfile.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_width = orig_height
    out_height = orig_width

    output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".webm").name
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (out_width, out_height))

    elbow_angles, shoulder_angles, knee_angles, trunk_tilts = [], [], [], []
    frame_idx = 0

    with PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((frame_idx / fps) * 1000)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks[0]

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

                elbow_angles.append(elbow_angle)
                shoulder_angles.append(shoulder_angle)
                knee_angles.append(knee_angle)
                trunk_tilts.append(trunk_tilt)

                cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), (0, 255, 0), 3)
                cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), (0, 255, 0), 3)
                cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 0, 0), 3)
                cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankle[0]), int(ankle[1])), (255, 0, 0), 3)

                for lm in [shoulder, elbow, wrist, hip, knee, ankle]:
                    cv2.circle(frame, (int(lm[0]), int(lm[1])), 6, (0, 0, 255), -1)

            out.write(frame)
            frame_idx += 1

    cap.release()
    out.release()
    os.remove(tfile.name)

    peaks = {
        "elbow": int(np.percentile(elbow_angles, 90)) if elbow_angles else 0,
        "shoulder": int(np.percentile(shoulder_angles, 90)) if shoulder_angles else 0,
        "knee": int(np.percentile(knee_angles, 90)) if knee_angles else 0,
        "trunk": int(np.percentile(trunk_tilts, 90)) if trunk_tilts else 0,
    }

    return output_video_path, peaks

# -----------------------------------------------------------------------------
# 3. Streamlit Interface & Multi-Video Logic
# -----------------------------------------------------------------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload Pitching Videos (1 or Multiple)", 
    type=["mp4", "mov", "avi"], 
    accept_multiple_files=True
)

if uploaded_files:
    session_peaks = {"elbow": [], "shoulder": [], "knee": [], "trunk": []}
    processed_videos = []

    progress_bar = st.progress(0, text="Processing video session...")
    
    for idx, uploaded_file in enumerate(uploaded_files):
        out_path, peaks = process_single_video(uploaded_file.read(), uploaded_file.name)
        processed_videos.append((uploaded_file.name, out_path, peaks))
        
        session_peaks["elbow"].append(peaks["elbow"])
        session_peaks["shoulder"].append(peaks["shoulder"])
        session_peaks["knee"].append(peaks["knee"])
        session_peaks["trunk"].append(peaks["trunk"])

        progress_bar.progress(int(((idx + 1) / len(uploaded_files)) * 100))

    progress_bar.empty()

    # Layout Output
    col_video, col_summary = st.columns([1, 1])

    with col_video:
        st.subheader("📺 Processed Video Clips")
        selected_vid_name = st.selectbox(
            "Select clip to inspect:", 
            options=[vid[0] for vid in processed_videos]
        )
        # Find matching video path
        selected_vid = next(item for item in processed_videos if item[0] == selected_vid_name)
        st.video(selected_vid[1])

    with col_summary:
        st.subheader("📊 Pitcher Baseline & Range Profile")
        st.caption(f"Calculated across {len(uploaded_files)} uploaded video sample(s).")

        # Summary Table View
        def format_range_col(data_list):
            if not data_list:
                return "N/A"
            min_val, max_val = min(data_list), max(data_list)
            avg_val = int(np.mean(data_list))
            if len(data_list) == 1:
                return f"{avg_val}°"
            return f"**{min_val}° – {max_val}°** *(Avg: {avg_val}°)*"

        summary_data = [
            {
                "Joint Metric": "Max Elbow Extension",
                "Pitcher Range (Session)": format_range_col(session_peaks["elbow"]),
                "Benchmark Range": "160° – 180°"
            },
            {
                "Joint Metric": "Max Shoulder Abduction",
                "Pitcher Range (Session)": format_range_col(session_peaks["shoulder"]),
                "Benchmark Range": "85° – 100°"
            },
            {
                "Joint Metric": "Max Lead Knee Extension",
                "Pitcher Range (Session)": format_range_col(session_peaks["knee"]),
                "Benchmark Range": "160° – 180°"
            },
            {
                "Joint Metric": "Max Trunk Forward Tilt",
                "Pitcher Range (Session)": format_range_col(session_peaks["trunk"]),
                "Benchmark Range": "30° – 50°"
            },
        ]

        st.table(summary_data)

else:
    st.info("👈 Please upload 1 or more pitching videos from the sidebar to establish a biomechanical range profile.")