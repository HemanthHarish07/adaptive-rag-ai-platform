import os
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# Import lazily inside lifespan so uvicorn import does not fail due to
# environment/PYTHONPATH issues.
IntentClassifier = None
classifier = None


def _load_classifier():
    global IntentClassifier
    if IntentClassifier is None:
        from src.inference import IntentClassifier as _IC

        IntentClassifier = _IC

    model_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "models", "intent_classifier")
    )
    return IntentClassifier(model_dir)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    try:
        # NOTE: explicit sys.path insertion here to avoid platform/subprocess
        # issues. This is not a "relative-import hack" in app code; it only
        # ensures the local src package is importable when uvicorn runs.
        import sys
        src_pkg_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
        if src_pkg_root not in sys.path:
            sys.path.insert(0, src_pkg_root)

        classifier = _load_classifier()

        print("Model loaded successfully at startup!")
    except Exception as e:
        print(f"Warning: Could not load model at startup: {e}")
        print("FastAPI app will start but `/predict` will raise errors until model is trained.")
    yield
    print("Shutting down application...")


app = FastAPI(
    title="Educational Intent Classifier API",
    description="Production FastAPI service for classifying student intent in a RAG routing pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str = Field(
        ...,
        description="The student educational query to classify",
        example="How do I write a binary search tree in Python?",
    )


class BatchQueryRequest(BaseModel):
    queries: List[str] = Field(
        ...,
        description="List of student queries to classify",
        example=["What is mitosis?", "Explain how quicksort works."],
    )


class PredictionResponse(BaseModel):
    query: str
    label: str
    confidence: float
    probabilities: Dict[str, float]


class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_dashboard():
    return HTMLResponse("<html><body><h3>EduIntentClassifier is running.</h3></body></html>")


@app.get("/health", tags=["System"])
async def health_check():
    if classifier is None:
        model_status = "not_loaded"
    elif getattr(classifier, "use_fallback", False):
        model_status = "fallback"
    else:
        model_status = "loaded"

    return {
        "status": "healthy",
        "model_status": model_status,
        "device": getattr(classifier, "device", "unknown"),
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_intent(request: QueryRequest):
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure that the fine-tuned model exists in the `models/intent_classifier` directory.",
        )

    try:
        result = classifier.predict(request.query)
        return PredictionResponse(
            query=result["query"],
            label=result["label"],
            confidence=result["confidence"],
            probabilities=result["probabilities"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Inference"])
async def predict_intent_batch(request: BatchQueryRequest):
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Ensure that the fine-tuned model exists in the `models/intent_classifier` directory.",
        )

    try:
        results = classifier.predict_batch(request.queries)
        return BatchPredictionResponse(
            predictions=[
                PredictionResponse(
                    query=res["query"],
                    label=res["label"],
                    confidence=res["confidence"],
                    probabilities=res["probabilities"],
                )
                for res in results
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

