from pathlib import Path

import numpy as np
import onnxruntime as ort

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "fraud_model.onnx"
N_FEATURES = 30

_session = ort.InferenceSession(str(MODEL_PATH))
_input_name = _session.get_inputs()[0].name


def score(features: list[float]) -> float:
    x = np.asarray([features], dtype=np.float32)
    return float(_session.run(None, {_input_name: x})[1][0][1])
