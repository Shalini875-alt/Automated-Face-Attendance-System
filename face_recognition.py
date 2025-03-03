import cv2
import os
import numpy as np
import pandas as pd
from datetime import datetime

# Load OpenCV LBPH Face Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces_folder = "faces"
known_face_encodings = []
known_face_names = []
labels = []
label_id = 0

# Load training data
for filename in os.listdir(faces_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(faces_folder, filename)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        known_face_encodings.append(image)
        known_face_names.append(os.path.splitext(filename)[0])  # Remove file extension
        labels.append(label_id)
        label_id += 1

# Train LBPH recognizer
if known_face_encodings:
    recognizer.train(known_face_encodings, np.array(labels))

# Load OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Open webcam
cap = cv2.VideoCapture(0)

# Create or load attendance.csv
csv_file = "attendance.csv"
if not os.path.exists(csv_file):
    df = pd.DataFrame(columns=["Name", "Time"])
    df.to_csv(csv_file, index=False)

print("Face recognition started. Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.2, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y+h, x:x+w]

        # Recognize face using LBPH
        label, confidence = recognizer.predict(face_roi)
        name = "Unknown"
        if confidence < 70:  # Lower confidence means better match
            name = known_face_names[label]

            # Mark attendance
            df = pd.read_csv(csv_file)
            if name not in df["Name"].values:
                new_entry = pd.DataFrame([[name, datetime.now().strftime("%Y-%m-%d %H:%M:%S")]], columns=["Name", "Time"])
                df = pd.concat([df, new_entry], ignore_index=True)
                df.to_csv(csv_file, index=False)

        # Draw box and label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
