import logging
from contextlib import asynccontextmanager
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_fastapi_instrumentator import Instrumentator
from src.config import MODEL_PATH
from src.data import FEATURES

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
)
logger = logging.getLogger("heart-api")
model = None


class Patient(BaseModel):
    age: float = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: float = Field(ge=50, le=300)
    chol: float = Field(ge=50, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: float = Field(ge=40, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    if not MODEL_PATH.exists():
        raise RuntimeError("Model missing. Run: python src/train.py")
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded from %s", MODEL_PATH)
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)
Instrumentator().instrument(app).expose(app, endpoint="/metrics")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(patient: Patient):
    if model is None:
        raise HTTPException(503, "Model not loaded")
    row = pd.DataFrame([[getattr(patient, f) for f in FEATURES]], columns=FEATURES)
    confidence = float(model.predict_proba(row)[0, 1])
    prediction = int(confidence >= 0.5)
    logger.info("prediction=%s confidence=%.4f", prediction, confidence)
    return {
        "prediction": prediction,
        "risk_label": "higher_risk" if prediction else "lower_risk",
        "confidence": round(confidence, 4),
    }
