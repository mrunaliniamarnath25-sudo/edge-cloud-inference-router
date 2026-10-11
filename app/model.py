import os
from pathlib import Path

import numpy as np
import onnxruntime as ort

N_FEATURES = 30
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class Scorer:
    def __init__(self, env_var: str, local_name: str):
        self.env_var = env_var
        self.local_name = local_name
        self.path = None
        self._session = None
        self._input = None

    def _load(self):
        handle = os.getenv(self.env_var)
        if handle:
            import kagglehub

            folder = Path(kagglehub.model_download(handle))
            self.path = next(folder.rglob("*.onnx"))
        else:
            self.path = MODELS_DIR / self.local_name
        self._session = ort.InferenceSession(str(self.path))
        self._input = self._session.get_inputs()[0].name

    def __call__(self, features: list[float]) -> float:
        if self._session is None:
            self._load()
        x = np.asarray([features], dtype=np.float32)
        return float(self._session.run(None, {self._input: x})[1][0][1])


score = Scorer("MODEL_HANDLE", "fraud_model.onnx")
cloud_score = Scorer("CLOUD_MODEL_HANDLE", "fraud_model_cloud.onnx")
