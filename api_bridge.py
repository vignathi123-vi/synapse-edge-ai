from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

app = FastAPI(
    title="Synapse Vehicle Data Bridge",
    description="Team Flint - Edge AI for Personalized and Connected Vehicles",
)

try:
    saved = joblib.load("style_model.joblib")
    model = saved["model"]
    feature_cols = saved["feature_cols"]
except FileNotFoundError:
    model = None
    feature_cols = []
    print("style_model.joblib not found, run train_style_model.py first")


class CANStylePayload(BaseModel):
    acc_x: float
    acc_y: float
    acc_z: float
    gyro_x: float
    gyro_y: float
    gyro_z: float
    vehicle_id: str


def build_features(payload: CANStylePayload):
    acc_mag = (payload.acc_x**2 + payload.acc_y**2 + payload.acc_z**2) ** 0.5
    gyro_mag = (payload.gyro_x**2 + payload.gyro_y**2 + payload.gyro_z**2) ** 0.5

    base = {
        "AccX": payload.acc_x, "AccY": payload.acc_y, "AccZ": payload.acc_z,
        "GyroX": payload.gyro_x, "GyroY": payload.gyro_y, "GyroZ": payload.gyro_z,
        "acc_magnitude": acc_mag, "gyro_magnitude": gyro_mag,
    }
    for col, val in list(base.items()):
        base[f"{col}_roll_mean"] = val
        base[f"{col}_roll_std"] = abs(val) * 0.15

    row = [base.get(col, 0.0) for col in feature_cols]
    return np.array(row).reshape(1, -1)


def style_to_preferences(style: str):
    mapping = {
        "AGGRESSIVE": {"driving_mode": "Sport", "climate_temp_c": 19, "music_genre": "Rock"},
        "NORMAL": {"driving_mode": "Comfort", "climate_temp_c": 22, "music_genre": "Pop"},
        "SLOW": {"driving_mode": "Eco", "climate_temp_c": 23, "music_genre": "Lo-fi / Focus"},
    }
    return mapping.get(style, mapping["NORMAL"])


@app.post("/bridge/vehicle-data")
def bridge_vehicle_data(payload: CANStylePayload):
    if model is None:
        return {"error": "model not loaded"}

    features = build_features(payload)
    predicted_style = model.predict(features)[0]
    preferences = style_to_preferences(predicted_style)

    cloud_format = {
        "vehicleId": payload.vehicle_id,
        "telemetry": {
            "drivingStyle": predicted_style,
            "recommendedSettings": preferences,
        },
        "source": "Synapse-OnDeviceModel",
        "schemaVersion": "cloud-telematics-v1",
    }
    return cloud_format


@app.get("/")
def root():
    return {"status": "Synapse bridge running, visit /docs to test"}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
