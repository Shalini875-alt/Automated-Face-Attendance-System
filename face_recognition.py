import cv2
import face_recognition
import numpy as np

def recognize_face():
    # Load a sample picture and learn how to recognize it.
    known_image = face_recognition.load_image_file("known_faces/known.jpg")
    known_encoding = face_recognition.face_encodings(known_image)[0]

    # Initialize some variables
    video_capture = cv2.VideoCapture(0)

    process_this_frame = True
    face_recognized = False

    while True:
        ret, frame = video_capture.read()
        
        if not ret:
            return "Failed to open camera."

        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]

            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces([known_encoding], face_encoding)
                face_distances = face_recognition.face_distance([known_encoding], face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    face_recognized = True
                    break

        if face_recognized:
            break

    video_capture.release()
    cv2.destroyAllWindows()

    if face_recognized:
        return "Face recognized successfully!"
    else:
        return "No matching face found."



