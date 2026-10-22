import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import time

CSV_PATH = "train_motion_data.csv"


def find_column(columns, candidates):
    lower_map = {c.lower(): c for c in columns}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None


def main():
    df = pd.read_csv(CSV_PATH)

    acc_x = find_column(df.columns, ["AccX"])
    acc_y = find_column(df.columns, ["AccY"])
    acc_z = find_column(df.columns, ["AccZ"])
    gyro_x = find_column(df.columns, ["GyroX"])
    gyro_y = find_column(df.columns, ["GyroY"])
    gyro_z = find_column(df.columns, ["GyroZ"])
    label_col = find_column(df.columns, ["Class"])

    feature_cols = [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    df = df.dropna(subset=feature_cols + [label_col])

    df["acc_magnitude"] = (df[acc_x]**2 + df[acc_y]**2 + df[acc_z]**2) ** 0.5
    df["gyro_magnitude"] = (df[gyro_x]**2 + df[gyro_y]**2 + df[gyro_z]**2) ** 0.5

    window = 15
    for col in feature_cols + ["acc_magnitude", "gyro_magnitude"]:
        df[f"{col}_roll_mean"] = df[col].rolling(window, min_periods=1).mean()
        df[f"{col}_roll_std"] = df[col].rolling(window, min_periods=1).std().fillna(0)

    engineered_cols = [c for c in df.columns if c.endswith("_roll_mean") or c.endswith("_roll_std")]
    all_features = feature_cols + ["acc_magnitude", "gyro_magnitude"] + engineered_cols

    X = df[all_features]
    y = df[label_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (RBF kernel)": SVC(kernel="rbf"),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results = []
    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0

        t0 = time.time()
        preds = model.predict(X_test)
        infer_time_ms = ((time.time() - t0) / len(X_test)) * 1000

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="macro")

        results.append({
            "model": name,
            "accuracy": acc,
            "f1_macro": f1,
            "train_time_s": train_time,
            "avg_inference_ms": infer_time_ms,
        })

    results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
    print("\nModel comparison results:\n")
    print(results_df.to_string(index=False))

    results_df.to_csv("model_comparison_results.csv", index=False)
    print("\nSaved results to model_comparison_results.csv")
    print("Screenshot this table for your slides/proof of comparison.")


if __name__ == "__main__":
    main()
