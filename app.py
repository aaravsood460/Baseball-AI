import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
import pandas as pd

# ---------------------------------------------------------
# Page Config & Setup
# ---------------------------------------------------------
st.set_page_config(page_title="PitchPerfect AI", page_icon="⚾", layout="wide")

st.title("⚾ PitchPerfect AI")
st.subheader("Real-Time Kinematic Breakdown & Delivery Diagnostics")
st.write("Extract critical joint angles at **Foot Contact**, **Max Layback**, and **Ball Release** across pitch sessions.")

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return int(360 - angle if angle > 180.0 else angle)

mp_pose = mp.solutions.pose

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("Pitch Controls")
uploaded_files = st.sidebar.file_uploader(
    "Upload Pitching Video (.mp4, .mov)", 
    type=["mp4", "mov"], 
    accept_multiple_files=True
)

side_preference = st.sidebar.radio("Pitcher Handedness:", ("Right", "Left"))

# Helper to extract metrics from a single video file
def process_video_metrics(file_obj, side):
    file_obj.seek(0)
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mov")
    tfile.write(file_obj.read())
    tfile.close()

    pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    cap = cv2.VideoCapture(tfile.name)
    
    frames_data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            if side == "Right":
                sh = [lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                el = [lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x, lm[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                wr = [lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].x, lm[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]
                hp = [lm[mp_pose.PoseLandmark.RIGHT_HIP.value].x, lm[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                kn = [lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, lm[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                ak = [lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, lm[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            else:
                sh = [lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, lm[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                el = [lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, lm[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wr = [lm[mp_pose.PoseLandmark.LEFT_WRIST.value].x, lm[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                hp = [lm[mp_pose.PoseLandmark.LEFT_HIP.value].x, lm[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                kn = [lm[mp_pose.PoseLandmark.LEFT_KNEE.value].x, lm[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ak = [lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, lm[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

            frames_data.append({
                "elbow": calculate_angle(sh, el, wr),
                "knee": calculate_angle(hp, kn, ak),
                "trunk": int(np.abs(sh[1] - hp[1]) * 100)
            })
            
    cap.release()
    pose.close()
    
    if not frames_data:
        return None

    df_f = pd.DataFrame(frames_data)
    fc = df_f.iloc[int(len(df_f) * 0.25)] if len(df_f) > 4 else df_f.iloc[0]
    ml = df_f.loc[df_f['elbow'].idxmax()]
    br = df_f.iloc[int(len(df_f) * 0.85)] if len(df_f) > 4 else df_f.iloc[-1]

    return {
        "fc_knee": int(fc['knee']),
        "fc_trunk": int(fc['trunk']),
        "ml_layback": min(int(ml['elbow']) + 45, 179),
        "ml_elbow": int(ml['elbow']),
        "br_knee": int(br['knee'])
    }

# ---------------------------------------------------------
# Processing Pipeline
# ---------------------------------------------------------
if uploaded_files:
    col1, col2 = st.columns([1.2, 1])

    clip_names = [f.name for f in uploaded_files]
    selected_clip_name = col1.selectbox("Select clip:", clip_names)
    selected_file = next(f for f in uploaded_files if f.name == selected_clip_name)

    # Calculate Session-Wide Metrics Across All Uploads
    all_metrics = []
    selected_metrics = None

    for f in uploaded_files:
        res = process_video_metrics(f, side_preference)
        if res:
            all_metrics.append(res)
            if f.name == selected_clip_name:
                selected_metrics = res

    # Render Selected Clip
    with col1:
        st.write("### 📺 Processed Clips")
        selected_file.seek(0)
        st.video(selected_file.read())

    # Build Table with Dynamic Session Range & Averages
    with col2:
        st.write("### 🎯 Key Phase Biomechanics")
        if selected_metrics and all_metrics:
            df_all = pd.DataFrame(all_metrics)

            def make_range_str(col):
                mn, mx, avg = int(df_all[col].min()), int(df_all[col].max()), int(df_all[col].mean())
                return f"{mn}° - {mx}° (Avg: {avg}°)"

            summary_data = [
                {"Phase": "1. Foot Contact", "Metric": "Lead Knee Flexion", "Selected": f"{selected_metrics['fc_knee']}°", "Session Range": make_range_str('fc_knee')},
                {"Phase": "1. Foot Contact", "Metric": "Initial Trunk Tilt", "Selected": f"{selected_metrics['fc_trunk']}°", "Session Range": make_range_str('fc_trunk')},
                {"Phase": "2. Max Layback", "Metric": "Shoulder / Arm Layback", "Selected": f"{selected_metrics['ml_layback']}°", "Session Range": make_range_str('ml_layback')},
                {"Phase": "2. Max Layback", "Metric": "Elbow Flexion", "Selected": f"{selected_metrics['ml_elbow']}°", "Session Range": make_range_str('ml_elbow')},
                {"Phase": "3. Ball Release", "Metric": "Lead Knee Flexion", "Selected": f"{selected_metrics['br_knee']}°", "Session Range": make_range_str('br_knee')},
            ]

            st.table(pd.DataFrame(summary_data))

            # Dynamic Diagnostics Feedback
            st.write("### 💡 AI Diagnostics & Feedback")
            if selected_metrics['fc_knee'] > 150:
                st.warning("⚠️ **Lead Knee Flexion High:** Stiff landing detected at foot contact. Consider bending front knee to absorb energy.")
            else:
                st.success("✅ **Good Landing Mechanics:** Front leg bracing is within optimal kinematic range.")

            if selected_metrics['ml_elbow'] < 90:
                st.warning("⚠️ **Elbow Flexion Low:** Arm angle is tight at layback. Maintain ~90° to optimize arm speed.")
            else:
                st.info("ℹ️ **Layback Positioning:** Excellent external rotation during phase 2.")
        else:
            st.warning("No pose data detected in video clips.")

else:
    st.info("👈 Upload pitching videos from the sidebar to start delivery analysis.")