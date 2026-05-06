import streamlit as st
import cv2
import numpy as np
import os
import datetime
from collections import deque

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Face Dashboard", layout="wide")
st.title("🧠 AI Face Recognition Dashboard")

# ---------------- LOAD FACE DATA ----------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

known_faces = []
known_names = []

folder = "known_faces"

if not os.path.exists(folder):
    os.makedirs(folder)

for file in os.listdir(folder):
    img = cv2.imread(os.path.join(folder, file), 0)
    if img is not None:
        img = cv2.resize(img, (100, 100))
        known_faces.append(img)
        known_names.append(os.path.splitext(file)[0])

# ---------------- SESSION ----------------
if "log" not in st.session_state:
    st.session_state.log = deque(maxlen=8)

# ---------------- SIDEBAR MODE ----------------
mode = st.sidebar.radio("Choose Input:", ["📷 Webcam", "🖼 Upload Image"])

face_count = st.sidebar.empty()
status_box = st.sidebar.empty()

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2, 1])
frame_box = col1.empty()
info_box = col2.container()

# ---------------- PROCESS FUNCTION ----------------
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    detected = []

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))

        name = "Unknown"
        min_diff = 99999

        for i, known in enumerate(known_faces):
            diff = np.sum((known - face) ** 2)
            if diff < min_diff:
                min_diff = diff
                name = known_names[i]

        detected.append(name)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        log = f"{name} @ {datetime.datetime.now().strftime('%H:%M:%S')}"
        st.session_state.log.appendleft(log)

        with open("attendance.txt", "a") as f:
            f.write(log + "\n")

    face_count.metric("Faces Detected", len(faces))

    if len(faces) > 0:
        status_box.success("FACE DETECTED")
    else:
        status_box.error("NO FACE DETECTED")

    return frame, detected

# ---------------- WEBCAM ----------------
if mode == "📷 Webcam":

    run = st.button("▶ Capture Frame")
    cap = cv2.VideoCapture(0)

    if run:
        ret, frame = cap.read()

        if ret:
            frame = cv2.resize(frame, (700, 500))
            frame, detected = detect_faces(frame)

            frame_box.image(frame, channels="BGR")

            with info_box:
                st.subheader("📋 Detection Panel")
                for d in detected:
                    st.write(f"✔ {d}")

    cap.release()

# ---------------- IMAGE UPLOAD ----------------
elif mode == "🖼 Upload Image":

    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

        frame = cv2.resize(frame, (700, 500))
        frame, detected = detect_faces(frame)

        frame_box.image(frame, channels="BGR")

        with info_box:
            st.subheader("📋 Detection Panel")
            for d in detected:
                st.write(f"✔ {d}")

# ---------------- LOG ----------------
st.subheader("📜 Recent Activity")
for item in st.session_state.log:
    st.write(item)