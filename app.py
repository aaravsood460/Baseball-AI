import os
import tempfile
import urllib.request
import cv2
import numpy as np
import streamlit as st
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Streamlit config
st.set_page_config(page_title="AI Pitching Mechanics Analyzer", page_icon="⚾", layout="wide")

st.title("⚾ AI Pitching Mechanics & Biomechanics Analyzer")
st.write("Extract critical joint angles at **Foot Contact**, **Max Layback**, and **Ball Release** across pitch sessions.")

# Cache model download so it doesn't redownload every run
@st.cache_resource
def load_pose_model():
    model_path = os.path.join(tempfile.gettempdir(), "pose_landmarker_full.task")
    if not os.path.exists(model_path) or os.path.getsize(model_path) < 1000000:
        url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
        urllib.request.urlretrieve(url, model_path)
    return model_path

# Angle math helper
def get_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    ang = np.abs(rad * 180.0 / np.pi)
    if ang > 180.0:
        ang = 360.0 - ang
    return int(ang)

def process_video(file_bytes):
    model_path = load_pose_model()
    
    base_opts = python.BaseOptions(model_asset_path=model_path)
    opts = vision.PoseLandmarkerOptions(base_options=base_opts, running_mode=vision.RunningMode.VIDEO)

    # Save uploaded bytes to temp file for OpenCV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(file_bytes)
        in_path = tmp.name

    cap = cv2.VideoCapture(in_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or np.isnan(fps):
        fps = 30.0 # Default fallback if fps fails

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 720
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1280

    out_file = tempfile.NamedTemporaryFile(delete=False, suffix=".webm")
    out_path = out_file.name
    out_file.close()

    # VP80 codec for browser compatibility in Streamlit
    fourcc = cv2.VideoWriter_fourcc(*'VP80')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frames_data = []
    frame_count = 0

    with vision.PoseLandmarker.create_from_options(opts) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            time_ms = int((frame_count / fps) * 1000)
            
            res = landmarker.detect_for_video(mp_img, time_ms)
            
            fdata = {"frame_idx": frame_count, "has_pose": False}

            if res.pose_landmarks and len(res.pose_landmarks) > 0:
                pts = res.pose_landmarks[0]

                # Key joints (MediaPipe pose indices)
                shldr = [pts[12].x * w, pts[12].y * h]
                elbw  = [pts[14].x * w, pts[14].y * h]
                wrst  = [pts[16].x * w, pts[16].y * h]
                hip   = [pts[24].x * w, pts[24].y * h]
                knee  = [pts[26].x * w, pts[26].y * h]
                ankl  = [pts[28].x * w, pts[28].y * h]

                # Angle calculations
                elbow_ang  = get_angle(shldr, elbw, wrst)
                knee_ang   = get_angle(hip, knee, ankl)
                shldr_ang  = get_angle(hip, shldr, elbw)
                trunk_tilt = 180 - get_angle(shldr, hip, knee)

                fdata.update({
                    "has_pose": True,
                    "elbow": elbow_ang,
                    "knee": knee_ang,
                    "shoulder": shldr_ang,
                    "trunk": trunk_tilt,
                    "wrist_x": wrst[0],
                    "ankle_y": ankl[1]
                })

                # Draw skeleton overlay
                cv2.line(frame, (int(shldr[0]), int(shldr[1])), (int(elbw[0]), int(elbw[1])), (0, 255, 0), 3)
                cv2.line(frame, (int(elbw[0]), int(elbw[1])), (int(wrst[0]), int(wrst[1])), (0, 255, 0), 3)
                cv2.line(frame, (int(hip[0]), int(hip[1])), (int(knee[0]), int(knee[1])), (255, 0, 0), 3)
                cv2.line(frame, (int(knee[0]), int(knee[1])), (int(ankl[0]), int(ankl[1])), (255, 0, 0), 3)

                for p in [shldr, elbw, wrst, hip, knee, ankl]:
                    cv2.circle(frame, (int(p[0]), int(p[1])), 6, (0, 0, 255), -1)

            out.write(frame)
            frames_data.append(fdata)
            frame_count += 1

    cap.release()
    out.release()
    if os.path.exists(in_path):
        os.remove(in_path)

    # Phase detection logic
    valid = [f for f in frames_data if f["has_pose"]]
    
    if len(valid) > 5:
        # Release frame = max forward wrist speed
        wrist_vels = [valid[i]["wrist_x"] - valid[i-1]["wrist_x"] for i in range(1, len(valid))]
        rel_idx = np.argmax(wrist_vels) + 1
        rel_f = valid[rel_idx]

        # Max Layback = peak shoulder abduction right before release
        pre_rel = valid[max(0, rel_idx - 15):rel_idx]
        lay_f = max(pre_rel, key=lambda x: x["shoulder"]) if pre_rel else rel_f

        # Foot Contact = lowest ankle position early in delivery
        early_f = valid[:max(1, rel_idx - 10)]
        fc_f = max(early_f, key=lambda x: x["ankle_y"]) if early_f else rel_f
    else:
        # Fallback if video is too short or pose lost
        fb = valid[0] if valid else {"elbow": 0, "shoulder": 0, "knee": 0, "trunk": 0}
        fc_f = lay_f = rel_f = fb

    phases = {
        "foot_contact": {"knee": fc_f.get("knee", 0), "trunk": fc_f.get("trunk", 0)},
        "max_layback":  {"shoulder": lay_f.get("shoulder", 0), "elbow": lay_f.get("elbow", 0)},
        "release":      {"knee": rel_f.get("knee", 0), "trunk": rel_f.get("trunk", 0), "elbow": rel_f.get("elbow", 0)}
    }

    return out_path, phases

def format_range(vals):
    if not vals:
        return "N/A"
    if len(vals) == 1:
        return f"{vals[0]}°"
    return f"{min(vals)}° – {max(vals)}° (Avg: {int(np.mean(vals))}°)"

# --- UI Setup ---
uploads = st.sidebar.file_uploader("Upload Pitching Videos", type=["mp4", "mov", "avi"], accept_multiple_files=True)

if uploads:
    processed = []
    pbar = st.progress(0, text="Processing pitching phases...")
    
    for i, file in enumerate(uploads):
        try:
            out_path, phases = process_video(file.read())
            processed.append((file.name, out_path, phases))
        except Exception as e:
            st.error(f"Error reading {file.name}: {e}")

        pbar.progress(int(((i + 1) / len(uploads)) * 100))

    pbar.empty()

    if processed:
        # Session metrics aggregation
        session = {
            "fc_knee":     [v[2]["foot_contact"]["knee"] for v in processed],
            "fc_trunk":    [v[2]["foot_contact"]["trunk"] for v in processed],
            "lb_shoulder": [v[2]["max_layback"]["shoulder"] for v in processed],
            "lb_elbow":    [v[2]["max_layback"]["elbow"] for v in processed],
            "rel_knee":    [v[2]["release"]["knee"] for v in processed],
            "rel_trunk":   [v[2]["release"]["trunk"] for v in processed],
        }

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📺 Processed Clips")
            selected_name = st.selectbox("Select clip:", options=[v[0] for v in processed])
            selected_vid = next(v for v in processed if v[0] == selected_name)
            st.video(selected_vid[1], format="video/webm")

        with col2:
            st.subheader("🎯 Key Phase Biomechanics")
            p_data = selected_vid[2]

            table_data = [
                {"Phase": "1. Foot Contact", "Metric": "Lead Knee Flexion", "Selected": f"{p_data['foot_contact']['knee']}°", "Session Range": format_range(session["fc_knee"]), "Benchmark": "130° – 150°"},
                {"Phase": "1. Foot Contact", "Metric": "Initial Trunk Tilt", "Selected": f"{p_data['foot_contact']['trunk']}°", "Session Range": format_range(session["fc_trunk"]), "Benchmark": "10° – 20°"},
                {"Phase": "2. Max Layback",  "Metric": "Shoulder Abduction", "Selected": f"{p_data['max_layback']['shoulder']}°", "Session Range": format_range(session["lb_shoulder"]), "Benchmark": "85° – 100°"},
                {"Phase": "2. Max Layback",  "Metric": "Elbow Flexion", "Selected": f"{p_data['max_layback']['elbow']}°", "Session Range": format_range(session["lb_elbow"]), "Benchmark": "80° – 105°"},
                {"Phase": "3. Ball Release", "Metric": "Lead Knee Extension", "Selected": f"{p_data['release']['knee']}°", "Session Range": format_range(session["rel_knee"]), "Benchmark": "160° – 180°"},
                {"Phase": "3. Ball Release", "Metric": "Forward Trunk Tilt", "Selected": f"{p_data['release']['trunk']}°", "Session Range": format_range(session["rel_trunk"]), "Benchmark": "35° – 50°"},
            ]
            st.table(table_data)

            st.markdown("---")
            st.subheader("💡 Automated Diagnostics")
            
            t1, t2 = st.tabs(["Selected Clip", "Overall Session"])

            with t1:
                # Clip feedback
                knee_rel = p_data["release"]["knee"]
                if knee_rel < 160:
                    st.warning(f"**⚠️ Soft Lead-Leg Block ({knee_rel}°)**\n\nFront leg isn't transferring max rotational force at release.\n\n👉 *Cue: Firm up on front heel at release.*")
                else:
                    st.success(f"**✅ Strong Lead-Leg Block ({knee_rel}°)**")

                shldr_lay = p_data["max_layback"]["shoulder"]
                if shldr_lay < 85:
                    st.warning(f"**⚠️ Low Arm Slot ({shldr_lay}°)**\n\nElbow is below shoulder line at layback, increasing joint stress.\n\n👉 *Cue: Keep elbow level with shoulder line.*")

                trunk_rel = p_data["release"]["trunk"]
                if trunk_rel < 35:
                    st.warning(f"**⚠️ Upright Finish ({trunk_rel}°)**\n\nCutting forward extension short puts extra load on arm deceleration.\n\n👉 *Cue: Drive chest over front knee.*")

            with t2:
                # Session feedback
                avg_knee = int(np.mean(session["rel_knee"]))
                if avg_knee < 160:
                    st.warning(f"**⚠️ Session Trend: Soft Lead Block (Avg: {avg_knee}°)**\n\nConsistent pattern across pitches. Work on lead-leg posting drills.")
                else:
                    st.success(f"**✅ Consistent Session Lead Block (Avg: {avg_knee}°)**")

                avg_shldr = int(np.mean(session["lb_shoulder"]))
                if avg_shldr < 85:
                    st.warning(f"**⚠️ Session Trend: Low Arm Slot (Avg: {avg_shldr}°)**\n\nRepeated low elbow slot increases cumulative stress.")

                avg_trunk = int(np.mean(session["rel_trunk"]))
                if avg_trunk < 35:
                    st.warning(f"**⚠️ Session Trend: Upright Finish (Avg: {avg_trunk}°)**\n\nConsistently cutting extension short. Focus on hip-hinge mobility.")

else:
    st.info("👈 Upload videos in sidebar to begin analysis.")