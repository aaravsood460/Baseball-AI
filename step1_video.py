import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- MATH FUNCTION FOR BIOMECHANICS ---
def calculate_angle(a, b, c):
    """Calculates the angle at point B given three coordinates A, B, and C."""
    a = np.array(a)  # First point
    b = np.array(b)  # Vertex point
    c = np.array(c)  # End point
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    
    angle = np.arccos(cosine_angle)
    return int(np.degrees(angle))


# 1. Setup the Modern MediaPipe Tasks API Engine
model_path = 'pose_landmarker_full.task'

PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
BaseOptions = python.BaseOptions

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO
)

POSE_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16), 
    (11, 23), (12, 24), (23, 24),                     
    (23, 25), (25, 27), (24, 26), (26, 28)            
]

with PoseLandmarker.create_from_options(options) as landmarker:
    video_path = "pitch.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open the video file.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video clip playback finished.")
            break

        h, w, _ = frame.shape
        timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = landmarker.detect_for_video(mp_image, timestamp_ms)

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

                # --- ADVANCED BIOMECHANICS DASHBOARD OVERLAY ---
                hud_y_offset = 40  # Starting vertical position for our dashboard text
                
                # 1. Right Elbow Angle (Shoulder 12 -> Elbow 14 -> Wrist 16)
                if 12 in points and 14 in points and 16 in points:
                    elbow_angle = calculate_angle(points[12], points[14], points[16])
                    cv2.putText(frame, f"Right Elbow: {elbow_angle} Deg", (20, hud_y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                    hud_y_offset += 30 # Drop down for the next line of text

                # 2. Right Shoulder Angle (Hip 24 -> Shoulder 12 -> Elbow 14)
                if 24 in points and 12 in points and 14 in points:
                    shoulder_angle = calculate_angle(points[24], points[12], points[14])
                    cv2.putText(frame, f"Right Shoulder: {shoulder_angle} Deg", (20, hud_y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2, cv2.LINE_AA)
                    hud_y_offset += 30

                # 3. Right Knee Angle (Hip 24 -> Knee 26 -> Ankle 28)
                if 24 in points and 26 in points and 28 in points:
                    knee_angle = calculate_angle(points[24], points[26], points[28])
                    cv2.putText(frame, f"Right Knee: {knee_angle} Deg", (20, hud_y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("AI Pitching Mechanics Tracker", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User manually closed the video stream.")
            break

    cap.release()
    cv2.destroyAllWindows()
