"""Pemuatan model dan inferensi. Pra-pemrosesan HARUS sama dengan notebook."""
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tensorflow import keras
from tensorflow.keras import layers

IMG_SIZE = 224

MODEL_FILES = {
    "efficientnetb0": "efficientnetb0_trashnet.h5",
    "baseline_cnn": "baseline_cnn_trashnet.h5",
}
MODEL_LABELS = {
    "efficientnetb0": "EfficientNetB0 (usulan)",
    "baseline_cnn": "CNN Baseline",
}


def load_h5_model(path):
    # Keras (.h5) tidak menghitung ulang mean/variance lapisan Normalization milik EfficientNet
    # saat memuat file -> wajib finalize_state() sebelum predict pertama.
    m = keras.models.load_model(path, compile=False)
    for l in m._flatten_layers(include_self=True, recursive=True):
        if isinstance(l, layers.Normalization):
            l.finalize_state()
    return m


def load_any(model_dir: Path, filename: str):
    h5 = model_dir / filename
    try:
        return load_h5_model(h5)
    except Exception as e:  # cadangan: format native .keras
        print(f"[WARN] gagal memuat {h5.name} ({e}); mencoba .keras")
        return keras.models.load_model(h5.with_suffix(".keras"), compile=False)


class Registry:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.class_names = json.loads((model_dir / "class_names.json").read_text())
        info_path = model_dir / "model_info.json"
        self.info = json.loads(info_path.read_text()) if info_path.exists() else {}
        self.models = {}
        for key, fname in MODEL_FILES.items():
            if (model_dir / fname).exists() or (model_dir / fname).with_suffix(".keras").exists():
                self.models[key] = load_any(model_dir, fname)
                print(f"[OK] model dimuat: {key}")
        if not self.models:
            raise RuntimeError(f"Tidak ada model ditemukan di {model_dir}")
        self.default = "efficientnetb0" if "efficientnetb0" in self.models else next(iter(self.models))
        self._warmup()

    def _warmup(self):
        x = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype="float32")
        for m in self.models.values():
            m.predict(x, verbose=0)

    def predict(self, img: Image.Image, key: str):
        # exif_transpose -> RGB -> resize 224 bicubic -> float32 0-255 (TANPA /255)
        img = ImageOps.exif_transpose(img).convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
        x = np.asarray(img, dtype="float32")[None]
        t0 = time.perf_counter()
        prob = self.models[key].predict(x, verbose=0)[0]
        ms = (time.perf_counter() - t0) * 1000
        order = np.argsort(-prob)
        return {
            "model": key,
            "prediction": self.class_names[int(order[0])],
            "confidence": float(prob[order[0]]),
            "probabilities": {self.class_names[i]: float(prob[i]) for i in order},
            "inference_ms": round(ms, 2),
        }
