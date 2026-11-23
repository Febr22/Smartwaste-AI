import io
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from inference import MODEL_LABELS, Registry

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "models"))
MAX_BYTES = 8 * 1024 * 1024
ALLOWED = {"image/jpeg", "image/png", "image/webp"}
ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["reg"] = Registry(MODEL_DIR)  # dimuat sekali saat startup
    yield
    state.clear()


app = FastAPI(title="Smartwaste-AI API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    reg = state["reg"]
    return {"status": "ok", "models": list(reg.models), "classes": reg.class_names}


@app.get("/models")
def models():
    reg = state["reg"]
    return {
        "default": reg.default,
        "models": [{"id": k, "label": MODEL_LABELS.get(k, k)} for k in reg.models],
        "info": reg.info.get("models", {}),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Query(None)):
    reg = state["reg"]
    key = model or reg.default
    if key not in reg.models:
        raise HTTPException(400, f"Model tidak dikenal: {key}. Pilihan: {list(reg.models)}")
    if file.content_type not in ALLOWED:
        raise HTTPException(415, "Format harus JPG, PNG, atau WEBP.")
    data = await file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Ukuran file maksimal 8 MB.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File bukan gambar yang valid.")
    return reg.predict(img, key)
