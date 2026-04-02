"""FastAPI model serving — prediction endpoint with input validation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pipeline.training import load_model

logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Pipeline API",
    description="Model serving API for the ML pipeline",
    version="1.0.0",
)

# Global model reference
_model: Any = None
_model_name: str = ""
_feature_names: list[str] = []


class PredictionRequest(BaseModel):
    """Input for prediction."""

    features: dict[str, float | int | str] = Field(..., description="Feature name-value pairs")


class PredictionResponse(BaseModel):
    """Output from prediction."""

    prediction: float | int | str
    probability: Optional[dict[str, float]] = None
    model: str
    feature_count: int


class BatchPredictionRequest(BaseModel):
    """Batch input for prediction."""

    instances: list[dict[str, float | int | str]] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    """Batch output from prediction."""

    predictions: list[float | int | str]
    count: int
    model: str


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Check API and model status."""
    return HealthResponse(
        status="healthy",
        model_loaded=_model is not None,
        model_name=_model_name,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    """Make a single prediction."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import pandas as pd

    try:
        df = pd.DataFrame([request.features])
        prediction = _model.predict(df)[0]

        probability = None
        if hasattr(_model, "predict_proba"):
            try:
                proba = _model.predict_proba(df)[0]
                classes = _model.classes_ if hasattr(_model, "classes_") else range(len(proba))
                probability = {str(c): round(float(p), 4) for c, p in zip(classes, proba)}
            except Exception:
                pass

        return PredictionResponse(
            prediction=prediction if not hasattr(prediction, "item") else prediction.item(),
            probability=probability,
            model=_model_name,
            feature_count=len(request.features),
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=422, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """Make batch predictions."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    import pandas as pd

    try:
        df = pd.DataFrame(request.instances)
        predictions = _model.predict(df).tolist()
        return BatchPredictionResponse(predictions=predictions, count=len(predictions), model=_model_name)
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/model/info")
def model_info() -> dict:
    """Get model metadata."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    info: dict[str, Any] = {
        "model_name": _model_name,
        "feature_names": _feature_names,
        "feature_count": len(_feature_names),
    }

    if hasattr(_model, "get_params"):
        info["params"] = _model.get_params()

    if hasattr(_model, "feature_importances_"):
        importances = _model.feature_importances_.tolist()
        if _feature_names:
            info["feature_importances"] = dict(zip(_feature_names, importances))

    return info


def load_serving_model(model_path: str, model_name: str = "", feature_names: list[str] | None = None) -> None:
    """Load a model for serving."""
    global _model, _model_name, _feature_names
    _model = load_model(model_path)
    _model_name = model_name or Path(model_path).stem
    _feature_names = feature_names or []
    logger.info(f"Serving model: {_model_name}")
