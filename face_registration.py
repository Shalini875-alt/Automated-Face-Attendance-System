import cv2
import os

def register_face():
    # Create a directory to save registered faces if it doesn't exist
    save_dir = 'known_faces'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        return "Failed to open camera."

    ret, frame = video_capture.read()

    if not ret:
        return "Failed to capture image from camera."

    # Save the captured image
    save_path = os.path.join(save_dir, 'known.jpg')
    cv2.imwrite(save_path, frame)

    video_capture.release()
    cv2.destroyAllWindows()

    return f"Face registered successfully and saved to {save_path}."

