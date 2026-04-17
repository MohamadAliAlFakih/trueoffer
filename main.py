# main.py
# FastAPI application — single POST /analyze endpoint
# Implements D-01 through D-07 from .planning/phases/03-fastapi-endpoint/03-CONTEXT.md

from contextlib import asynccontextmanager
from typing import Optional
import json
from pathlib import Path

import joblib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.chain import run_chain
from app.schemas import PredictionResponse, InsightResponse

ROOT_DIR = Path(__file__).resolve().parent


# D-01 + D-02: Request model — assumed_overrides optional, accepted but not yet passed
# to run_chain() (chain.py does not support overrides in Phase 3; wired in Phase 4)
class UnifiedRequest(BaseModel):
    message: str
    assumed_overrides: Optional[dict] = None


# D-05: Lifespan — verify critical resource files are present at startup.
# chain.py already loads _model and _stats at module-level (import time).
# This lifespan re-verifies them so that missing files produce a clear startup error
# rather than an opaque crash at the first request.
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = ROOT_DIR / "model" / "model.joblib"
    stats_path = ROOT_DIR / "data" / "stats.json"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Startup failed: model artifact not found at {model_path}. "
            "Run Phase 1 (ML Pipeline) to produce model/model.joblib."
        )
    if not stats_path.exists():
        raise FileNotFoundError(
            f"Startup failed: stats file not found at {stats_path}. "
            "Run Phase 1 (ML Pipeline) to produce data/stats.json."
        )
    # Load to confirm readability (not stored — chain.py owns the live copies)
    joblib.load(model_path)
    json.loads(stats_path.read_text())
    yield


app = FastAPI(lifespan=lifespan)

# D-06: CORS — allow_origins=["*"] for dev (Streamlit on 8501 -> FastAPI on 8000).
# NOTE: allow_credentials must NOT be set to True when allow_origins=["*"] —
# Starlette raises ValueError at startup if both are combined.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# D-03 + D-04 + D-07: Single endpoint — isinstance dispatch for {type, data} envelope.
@app.post("/analyze")
def analyze(req: UnifiedRequest) -> dict:
    if req.assumed_overrides:
        result = run_chain(req.message, assumed_overrides=req.assumed_overrides)
    else:
        result = run_chain(req.message)
    if isinstance(result, PredictionResponse):
        return {"type": "prediction", "data": result.model_dump()}
    return {"type": "insight", "data": result.model_dump()}
