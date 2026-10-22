import cv2
import json
import time
import os
import requests
import numpy as np

MODEL_PATH = "driver_model.yml"
LABELS_PATH = "labels.json"
PROFILES_PATH = "profiles.json"
V2X_FEED_PATH = "mock_v2x_feed.json"
BRIDGE_URL = "http://127.0.0.1:8000/bridge/vehicle-data"
BRIDGE_CALL_INTERVAL = 4

CONFIDENCE_THRESHOLD = 55
SMOOTHING_WINDOW = 15

WINDOW_TITLE = "Synapse - Team Flint | Personalized Connected Vehicle Demo"

# Brand colors (BGR format for OpenCV)
COLOR_ACCENT = (255, 210, 0)      # cyan-blue accent
COLOR_TEXT = (255, 255, 255)      # white
COLOR_HEADER = (11, 134, 184)     # brass/antique gold
COLOR_PANEL_BG = (30, 20, 10)     # dark navy panel

SENSOR_SCENARIOS = [
    {"label": "Calm cruising", "acc_x": 0.3, "acc_y": 0.1, "acc_z": 9.8, "gyro_x": 2.0, "gyro_y": 1.0, "gyro_z": 1.5},
    {"label": "Normal city driving", "acc_x": 2.0, "acc_y": -1.0, "acc_z": 9.8, "gyro_x": 10.0, "gyro_y": 5.0, "gyro_z": -6.0},
    {"label": "Sudden hard braking", "acc_x": 8.5, "acc_y": -6.2, "acc_z": 9.8, "gyro_x": 45.0, "gyro_y": 30.0, "gyro_z": -20.0},
]


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, "r") as f:
        return json.load(f)


def get_v2x_recommendation(v2x):
    tips = []
    if v2x.get("weather") == "Rain":
        tips.append("Rain detected nearby: switching to Eco + wipers suggested")
    if v2x.get("traffic_density") == "High":
        tips.append("Heavy traffic ahead: adaptive cruise recommended")
    if v2x.get("nearby_incident"):
        tips.append(f"Alert: {v2x['nearby_incident']}")
    return tips if tips else ["Conditions normal"]


def call_bridge(scenario, vehicle_id="FLINT_SYNAPSE_CAR_01"):
    payload = {
        "acc_x": scenario["acc_x"], "acc_y": scenario["acc_y"], "acc_z": scenario["acc_z"],
        "gyro_x": scenario["gyro_x"], "gyro_y": scenario["gyro_y"], "gyro_z": scenario["gyro_z"],
        "vehicle_id": vehicle_id,
    }
    try:
        resp = requests.post(BRIDGE_URL, json=payload, timeout=1.5)
        if resp.status_code == 200:
            return resp.json()
    except requests.exceptions.RequestException:
        pass
    return None


def draw_overlay(frame, lines, header="SYNAPSE"):
    cv2.putText(frame, header, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, COLOR_HEADER, 2, cv2.LINE_AA)

    y0 = 66
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (14, y0 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 2, cv2.LINE_AA)


def main():
    if not os.path.exists(MODEL_PATH):
        print("no trained model found, run capture_faces.py then train_model.py first")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_PATH)

    label_map = load_json(LABELS_PATH, {})
    profiles = load_json(PROFILES_PATH, {})
    v2x = load_json(V2X_FEED_PATH, {})
    v2x_tips = get_v2x_recommendation(v2x)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("could not open webcam")
        return

    recent_guesses = []
    stable_driver = "Guest"

    last_bridge_call = 0
    scenario_index = 0
    bridge_result = None
    bridge_status = "connecting to bridge..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        if now - last_bridge_call > BRIDGE_CALL_INTERVAL:
            scenario = SENSOR_SCENARIOS[scenario_index % len(SENSOR_SCENARIOS)]
            result = call_bridge(scenario)
            if result:
                bridge_result = result
                bridge_status = f"Bridge OK ({scenario['label']})"
            else:
                bridge_status = "Bridge OFFLINE - run api_bridge.py"
            scenario_index += 1
            last_bridge_call = now

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        driver_name = stable_driver
        for (x, y, w, h) in faces:
            face_img = cv2.resize(gray[y:y + h, x:x + w], (200, 200))
            label_id, confidence = recognizer.predict(face_img)

            if confidence < CONFIDENCE_THRESHOLD and str(label_id) in label_map:
                frame_guess = label_map[str(label_id)]
            else:
                frame_guess = "Guest"

            recent_guesses.append(frame_guess)
            if len(recent_guesses) > SMOOTHING_WINDOW:
                recent_guesses.pop(0)

            most_common = max(set(recent_guesses), key=recent_guesses.count)
            if recent_guesses.count(most_common) >= (SMOOTHING_WINDOW * 0.6):
                stable_driver = most_common

            driver_name = stable_driver
            cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_ACCENT, 2)
            cv2.putText(frame, driver_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2)
            break

        profile = profiles.get(driver_name, profiles.get("Guest", {}))

        lines = [
            f"Driver: {driver_name}",
            f"Seat: {profile.get('seat_position', '-')}",
            f"Climate: {profile.get('climate_temp_c', '-')} C",
            f"Music: {profile.get('music_genre', '-')}",
            f"Mode: {profile.get('driving_mode', '-')}",
        ] + [f"V2X: {tip}" for tip in v2x_tips]

        if bridge_result:
            telemetry = bridge_result.get("telemetry", {})
            lines.append(f"ML driving style: {telemetry.get('drivingStyle', '-')}")
            rec = telemetry.get("recommendedSettings", {})
            if rec:
                lines.append(f"Suggested mode: {rec.get('driving_mode', '-')}")
        lines.append(bridge_status)

        draw_overlay(frame, lines, header="SYNAPSE - Team Flint")

        cv2.imshow(WINDOW_TITLE, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
