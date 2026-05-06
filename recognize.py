import streamlit as st
import cv2
import numpy as np
import os
import datetime

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Face Dashboard", layout="wide")

st.title("🧠 AI Face Recognition Dashboard")

# ---------------- LOAD FACES ----------------
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

st.sidebar.header("📊 Status Panel")

run = st.checkbox("▶ Start Camera")

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

face_count_placeholder = st.sidebar.empty()
status_placeholder = st.sidebar.empty()

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not working")
        break

    frame = cv2.resize(frame, (700, 500))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_count_placeholder.write(f"👥 Faces Detected: {len(faces)}")

    status = "NO FACE"
    color = (0, 0, 255)

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

        # Draw box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Name label
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        status = "FACE DETECTED"
        color = (0, 255, 0)

        # Save attendance
        with open("attendance.txt", "a") as f:
            f.write(f"{name} - {datetime.datetime.now()}\n")

    # Status update
    status_placeholder.write(f"🔔 Status: {status}")

    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()