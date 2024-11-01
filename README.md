# ⚾ AI Pitching Mechanics & Biomechanics Analyzer

An interactive computer vision and biomechanics analysis application that tracks a pitcher's kinetic chain in real-time. Built using Python, MediaPipe, OpenCV, and Streamlit.

---

## 📌 Features
* Pose Estimation: Maps 33 key skeletal landmarks frame-by-frame using MediaPipe.
* Vector Biomechanics: Uses vector dot product mathematics and the Law of Cosines to calculate real-time joint angles for:
  * Right Elbow Flexion (Arm loading and release)
  * Right Shoulder Abduction (Trunk alignment and arm slot)
  * Right Lead Knee Flexion (Lower-body stability at foot plant)
* Web Interface: Interactive Streamlit GUI for video uploads and live analytics dashboards.

---

## 🛠️ Installation & Usage

1. Clone the repository:
   git clone https://github.com/YOUR_USERNAME/BaseballAI.git
   cd BaseballAI

2. Install dependencies:
   pip install -r requirements.txt

3. Launch the web application:
   streamlit run app.py

---

## 📐 Mathematical Framework
Joint angles are calculated between three spatial landmark vectors (BA and BC) using the inverse cosine of the vector dot product:

Angle = arccos( (Vector BA • Vector BC) / (|Vector BA| * |Vector BC|) )