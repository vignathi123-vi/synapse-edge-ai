import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib

CSV_PATH = "train_motion_data.csv"
MODEL_PATH = "style_model.joblib"


def find_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def main():
    df = pd.read_csv(CSV_PATH)
    print(list(df.columns))

    acc_x = find_column(df.columns, ["AccX", "acceleration_x", "Acc_X"])
    acc_y = find_column(df.columns, ["AccY", "acceleration_y", "Acc_Y"])
    acc_z = find_column(df.columns, ["AccZ", "acceleration_z", "Acc_Z"])
    gyro_x = find_column(df.columns, ["GyroX", "gyro_x", "Gyro_X"])
    gyro_y = find_column(df.columns, ["GyroY", "gyro_y", "Gyro_Y"])
    gyro_z = find_column(df.columns, ["GyroZ", "gyro_z", "Gyro_Z"])
    label_col = find_column(df.columns, ["Class", "label", "Behavior", "Driving_Behavior"])

    feature_cols = [c for c in [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z] if c is not None]

    df = df.dropna(subset=feature_cols + [label_col])

    # single readings are noisy, so add magnitude + rolling window features
    df["acc_magnitude"] = (df[acc_x]**2 + df[acc_y]**2 + df[acc_z]**2) ** 0.5
    df["gyro_magnitude"] = (df[gyro_x]**2 + df[gyro_y]**2 + df[gyro_z]**2) ** 0.5

    window = 15
    for col in feature_cols + ["acc_magnitude", "gyro_magnitude"]:
        df[f"{col}_roll_mean"] = df[col].rolling(window, min_periods=1).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window, min_periods=1).std().fillna(0)

    engineered_cols = [c for c in df.columns if c.endswith("_roll_mean") or c.endswith("_roll_std")]
    feature_cols = feature_cols + ["acc_magnitude", "gyro_magnitude"] + engineered_cols

    X = df[feature_cols]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"accuracy: {acc * 100:.1f}%")
    print(classification_report(y_test, preds))

    joblib.dump({"model": model, "feature_cols": feature_cols}, MODEL_PATH)
    print(f"saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
