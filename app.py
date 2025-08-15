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
st.write("Extract critical joint angles at **Foot Contact**, **Max Layback**, and **Ball Release**, with automated coaching feedback.")

# -----------------------------------------------------------------------------
# 1. Model Download & Cache
# -----------------------------------------------------------------------------
@st.cache_resource
def get_model_path():
    model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker_full.task")
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        urllib.request.urlretrieve(url, model_path)
    return model_path

def calculate_angle(a, b, c):
    """Calculates 2D interior angle (0-180 deg) between 3 joint points."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def process_single_video(file_bytes):
    """Processes video, detects key pitching phases, and outputs phase metrics."""
    model_path = get_model_path()
    
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(file_bytes)
        in_path = tfile.name

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0

    out_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if out_width == 0 or out_height == 0:
        out_width, out_height = 720, 1280

    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    out_path = out_file.name
    out_file.close()

    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(out_path, fourcc, fps, (out_width, out_height))

    frames_data = []
    frame_idx = 0

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            timestamp_ms = int((frame_idx / fps) * 1000)
            
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
            
            frame_metrics = {"frame": frame, "idx": frame_idx, "has_pose": False}

            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]

                shoulder = [landmarks[12].x * w, landmarks[12].y * h]
                elbow = [landmarks[14].x * w, landmarks[14].y * h]
                wrist = [landmarks[16].x * w, landmarks[16].y * h]
                hip = [landmarks[24].x * w, landmarks[24].y * h]
                knee = [landmarks[26].x * w, landmarks[26].y * h]
                ankle = [landmarks[28].x * w, landmarks[28].y * h]

                # Angles
                elbow_angle = int(calculate_angle(shoulder, elbow, wrist))
                knee_angle = int(calculate_angle(hip, knee, ankle))
                shoulder_angle = int(calculate_angle(hip, shoulder, elbow))
                trunk_tilt = int(180.0 - calculate_angle(shoulder, hip, knee))

                frame_metrics.update({
                    "has_pose": True,
                    "elbow": elbow_angle,
                    "knee": knee_angle,
                    "shoulder": shoulder_angle,
                    "trunk": trunk_tilt,
                    "wrist_x": wrist[0],
                    "ankle_y": ankle[1]
                })

                # Annotate skeleton
                cv2.line(frame, (int(shoulder[0]), int(shoulder[1])), (int(elbow[0]), int(elbow[1])), (0, 255, 0), 3)
                cv2.line(frame, (int(elbow[0]), int(elbow[1])), (int(wrist[0]), int(wrist[1])), (0, 255, 0), 3)
                cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 0, 0), 3)
                cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankle[0]), int(ankle[1])), (255, 0, 0), 3)

                for lm in [shoulder, elbow, wrist, hip, knee, ankle]:
                    cv2.circle(frame, (int(lm[0]), int(lm[1])), 6, (0, 0, 255), -1)

            out.write(frame)
            frames_data.append(frame_metrics)
            frame_idx += 1

    cap.release()
    out.release()
    if os.path.exists(in_path):
        os.remove(in_path)

    # -------------------------------------------------------------------------
    # Key Phase Identification Logic
    # -------------------------------------------------------------------------
    valid_frames = [f for f in frames_data if f["has_pose"]]
    
    if len(valid_frames) > 5:
        # Release point: Frame of peak forward wrist velocity
        wrist_velocities = [valid_frames[i]["wrist_x"] - valid_frames[i-1]["wrist_x"] for i in range(1, len(valid_frames))]
        release_idx = np.argmax(wrist_velocities) + 1
        release_frame = valid_frames[release_idx]

        # Max Layback: Occurs shortly before release point
        pre_release_frames = valid_frames[max(0, release_idx - 15):release_idx]
        layback_frame = max(pre_release_frames, key=lambda x: x["shoulder"]) if pre_release_frames else release_frame

        # Foot Contact: Occurs prior to layback
        early_frames = valid_frames[:max(1, release_idx - 10)]
        foot_contact_frame = max(early_frames, key=lambda x: x["ankle_y"]) if early_frames else release_frame
    else:
        default_f = valid_frames[0] if valid_frames else {"elbow": 0, "shoulder": 0, "knee": 0, "trunk": 0}
        foot_contact_frame = layback_frame = release_frame = default_f

    phase_metrics = {
        "foot_contact": {
            "knee": foot_contact_frame.get("knee", 0),
            "trunk": foot_contact_frame.get("trunk", 0)
        },
        "max_layback": {
            "shoulder": layback_frame.get("shoulder", 0),
            "elbow": layback_frame.get("elbow", 0)
        },
        "release": {
            "knee": release_frame.get("knee", 0),
            "trunk": release_frame.get("trunk", 0),
            "elbow": release_frame.get("elbow", 0)
        }
    }

    return out_path, phase_metrics

def generate_coaching_insights(p_data):
    """Generates automated coaching recommendations based on key phase angles."""
    alerts = []

    # 1. Lead Leg Block Check at Release
    lead_knee_rel = p_data["release"]["knee"]
    if lead_knee_rel < 160:
        alerts.append({
            "level": "warning",
            "title": "⚠️ Soft Lead-Leg Block",
            "msg": f"Lead knee extension at release is **{lead_knee_rel}°** (Benchmark: 160°–180°). The front leg is absorbing energy instead of driving rotational power into ball release.",
            "cue": "👉 **Coaching Cue:** Focus on 'posting up' firm on the front heel through release."
        })
    else:
        alerts.append({
            "level": "success",
            "title": "✅ Strong Lead-Leg Block",
            "msg": f"Lead knee extension at release is solid (**{lead_knee_rel}°**), transferring linear momentum effectively into rotational energy.",
            "cue": ""
        })

    # 2. Shoulder Abduction / Arm Slot Check at Layback
    shoulder_layback = p_data["max_layback"]["shoulder"]
    if shoulder_layback < 85:
        alerts.append({
            "level": "warning",
            "title": "⚠️ Dropped Elbow / Low Arm Slot",
            "msg": f"Shoulder abduction at max layback is **{shoulder_layback}°** (Benchmark: 85°–100°). A dropped elbow increases medial stress on the elbow joint.",
            "cue": "👉 **Coaching Cue:** Keep the elbow level with the shoulder line through arm cocking."
        })

    # 3. Forward Trunk Tilt Check at Release
    trunk_rel = p_data["release"]["trunk"]
    if trunk_rel < 30:
        alerts.append({
            "level": "warning",
            "title": "⚠️ Upright Finish",
            "msg": f"Forward trunk tilt at release is **{trunk_rel}°** (Benchmark: 35°–50°). Cutting extension short puts excess deceleration load on the throwing arm.",
            "cue": "👉 **Coaching Cue:** Drive the chest over the front knee at finish."
        })

    return alerts

# -----------------------------------------------------------------------------
# 2. Streamlit UI Logic
# -----------------------------------------------------------------------------
uploaded_files = st.sidebar.file_uploader(
    "Upload Pitching Videos (1 or Multiple)", 
    type=["mp4", "mov", "avi"], 
    accept_multiple_files=True
)

if uploaded_files:
    processed_videos = []

    progress_bar = st.progress(0, text="Processing pitching phases & generating analysis...")
    
    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            out_path, phases = process_single_video(uploaded_file.read())
            processed_videos.append((uploaded_file.name, out_path, phases))
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")

        progress_bar.progress(int(((idx + 1) / len(uploaded_files)) * 100))

    progress_bar.empty()

    if processed_videos:
        col_video, col_summary = st.columns([1, 1])

        with col_video:
            st.subheader("📺 Processed Video Clips")
            selected_vid_name = st.selectbox(
                "Select clip to inspect:", 
                options=[vid[0] for vid in processed_videos]
            )
            selected_vid = next(item for item in processed_videos if item[0] == selected_vid_name)
            st.video(selected_vid[1], format="video/webm")

        with col_summary:
            st.subheader("🎯 Key Phase Biomechanics")
            st.caption(f"Phase-isolated metrics for clip: **{selected_vid_name}**")

            p_data = selected_vid[2]

            phase_table = [
                {
                    "Pitching Phase": "1. Foot Contact (Stride Landing)",
                    "Metric": "Lead Knee Flexion",
                    "Measured": f"{p_data['foot_contact']['knee']}°",
                    "Benchmark": "130° – 150°"
                },
                {
                    "Pitching Phase": "1. Foot Contact (Stride Landing)",
                    "Metric": "Initial Trunk Tilt",
                    "Measured": f"{p_data['foot_contact']['trunk']}°",
                    "Benchmark": "10° – 20°"
                },
                {
                    "Pitching Phase": "2. Max Arm Layback",
                    "Metric": "Shoulder Abduction",
                    "Measured": f"{p_data['max_layback']['shoulder']}°",
                    "Benchmark": "85° – 100°"
                },
                {
                    "Pitching Phase": "2. Max Arm Layback",
                    "Metric": "Elbow Flexion",
                    "Measured": f"{p_data['max_layback']['elbow']}°",
                    "Benchmark": "80° – 105°"
                },
                {
                    "Pitching Phase": "3. Ball Release Point",
                    "Metric": "Lead Knee Extension (Block)",
                    "Measured": f"{p_data['release']['knee']}°",
                    "Benchmark": "160° – 180°"
                },
                {
                    "Pitching Phase": "3. Ball Release Point",
                    "Metric": "Forward Trunk Tilt",
                    "Measured": f"{p_data['release']['trunk']}°",
                    "Benchmark": "35° – 50°"
                },
            ]

            st.table(phase_table)

            # Automated Coaching Insights Section
            st.markdown("---")
            st.subheader("💡 Automated Mechanical Diagnostics")
            
            insights = generate_coaching_insights(p_data)
            for alert in insights:
                if alert["level"] == "warning":
                    st.warning(f"**{alert['title']}**\n\n{alert['msg']}\n\n{alert['cue']}")
                else:
                    st.success(f"**{alert['title']}**\n\n{alert['msg']}")

else:
    st.info("👈 Upload pitching videos to run automated key phase and mechanical diagnostics.")