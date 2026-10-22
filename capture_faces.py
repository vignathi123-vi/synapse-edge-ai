"""
capture_faces.py
-----------------
Step 1: Enroll driver profiles by capturing face images from your webcam.

Run this once per team member (or once per "test driver persona").
Usage:
    python capture_faces.py "Aditi"

It will open your webcam, detect your face, and save ~40 cropped face
images into dataset/Aditi/. Press 'q' to stop early.
"""

import cv2
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python capture_faces.py <driver_name>")
        sys.exit(1)

    driver_name = sys.argv[1]
    save_dir = os.path.join("dataset", driver_name)
    os.makedirs(save_dir, exist_ok=True)

    face_cascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        sys.exit(1)

    print(f"Capturing faces for '{driver_name}'. Look at the camera. Press 'q' to stop early.")

    count = 0
    max_images = 40

    while count < max_images:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            face_img = gray[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (200, 200))
            file_path = os.path.join(save_dir, f"{count}.jpg")
            cv2.imwrite(file_path, face_img)
            count += 1

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Captured: {count}/{max_images}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            break  # only take one face per frame

        cv2.imshow("Enrollment - press q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Done. Saved {count} images to {save_dir}")

if __name__ == "__main__":
    main()
