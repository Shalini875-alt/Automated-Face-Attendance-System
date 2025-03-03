import cv2
import os

# Create the 'faces' directory if it doesn't exist
if not os.path.exists("faces"):
    os.makedirs("faces")

# Initialize webcam
cap = cv2.VideoCapture(0)

# Ask for the person's name
name = input("Enter person's name: ")
image_path = f"faces/{name}.jpg"

print("Capturing face... Look at the camera.")

# Load face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]  # Crop the detected face
        cv2.imwrite(image_path, face)  # Save face image
        print(f"Face image saved as {image_path}")
        cap.release()
        cv2.destroyAllWindows()
        exit()

    cv2.imshow("Face Registration", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()