import joblib
import numpy as np
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort
import time

JOBLIB_PATH = "style_model.joblib"
ONNX_PATH = "style_model.onnx"


def main():
    saved = joblib.load(JOBLIB_PATH)
    model = saved["model"]
    feature_cols = saved["feature_cols"]
    n_features = len(feature_cols)

    initial_type = [("input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)

    with open(ONNX_PATH, "wb") as f:
        f.write(onnx_model.SerializeToString())

    print(f"saved onnx model to {ONNX_PATH}")

    session = ort.InferenceSession(ONNX_PATH)
    input_name = session.get_inputs()[0].name
    dummy_input = np.zeros((1, n_features), dtype=np.float32)

    t0 = time.time()
    result = session.run(None, {input_name: dummy_input})
    latency_ms = (time.time() - t0) * 1000

    print(f"predicted class: {result[0]}")
    print(f"inference latency: {latency_ms:.2f} ms")


if __name__ == "__main__":
    main()
