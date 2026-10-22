"""
train_model.py
---------------
Step 2: Train a local, on-device face recognition model (LBPH) on the
images you captured with capture_faces.py.

This model runs entirely on your laptop CPU -- no cloud, no GPU needed.
That's your "Edge AI" story: inference happens on-device.

Usage:
    python train_model.py
"""

import cv2
import os
import numpy as np
import json

DATASET_DIR = "dataset"
MODEL_PATH = "driver_model.yml"
LABELS_PATH = "labels.json"

def main():
    if not os.path.isdir(DATASET_DIR):
        print("No dataset/ folder found. Run capture_faces.py first for each driver.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []
    label_map = {}  # id -> name
    current_id = 0

    for driver_name in sorted(os.listdir(DATASET_DIR)):
        driver_dir = os.path.join(DATASET_DIR, driver_name)
        if not os.path.isdir(driver_dir):
            continue

        label_map[current_id] = driver_name

        for img_file in os.listdir(driver_dir):
            img_path = os.path.join(driver_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            faces.append(img)
            labels.append(current_id)

        current_id += 1

    if not faces:
        print("No face images found in dataset/. Run capture_faces.py first.")
        return

    print(f"Training on {len(faces)} images across {len(label_map)} driver profiles...")
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Labels saved to {LABELS_PATH}")
    print("Now run: python app.py")

if __name__ == "__main__":
    main()
