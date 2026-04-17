# app/chain.py
# 3-step chain: intent -> extraction -> prediction | insight

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import pandas as pd

from app.llm import GroqError, ParseError, call_groq, load_prompt
from app.schemas import (
    ExtractionResult,
    FeatureFlag,
    InsightResponse,
    IntentResult,
    LOCKED_FEATURES,
    PredictionResponse,
)

ROOT_DIR = Path(__file__).resolve().parent.parent

# --- Module-level resource loading ---

_model = joblib.load(ROOT_DIR / "model" / "model.joblib")
_stats = json.loads((ROOT_DIR / "data" / "stats.json").read_text())
_MEDIAN_PRICE: float = _stats["overall_price_stats"]["median"]

# --- Logging setup ---

_logs_dir = ROOT_DIR / "logs"
_logs_dir.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Internal helpers ---

def _log_run(variant: str, user_input: str, raw_output: str, validation_result: str) -> None:
    """Append one log entry to logs/chain.log."""
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "variant": variant,
        "input": user_input,
        "raw_output": raw_output,
        "validation_result": validation_result,
    }
    log_path = _logs_dir / "chain.log"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _classify_intent(user_message: str) -> str:
    """Return 'prediction' or 'insight'. Defaults to 'prediction' on any failure."""
    try:
        prompt = load_prompt("intent")
        raw = call_groq(prompt, user_message, json_mode=True)
        result = IntentResult.model_validate_json(raw)
        _log_run("intent-v1", user_message, raw, "valid")
        return result.intent
    except (GroqError, ParseError, Exception) as exc:
        _log_run("intent-v1", user_message, str(exc), "fallback:prediction")
        logger.warning("Intent classification failed, defaulting to prediction: %s", exc)
        return "prediction"


def _extract_features(user_message: str, variant: str = "extraction-v1") -> ExtractionResult:
    """
    Extract features from plain-English description.
    On any failure, returns ExtractionResult with all features ASSUMED from defaults.
    """
    prompt_name = "extraction_v2" if variant == "extraction-v2" else "extraction"
    try:
        prompt = load_prompt(prompt_name)
        raw = call_groq(prompt, user_message, json_mode=True)
        data = json.loads(raw)
        features = {
            k: FeatureFlag(value=v["value"], flag=v.get("flag", "ASSUMED"))
            for k, v in data.items()
            if k in LOCKED_FEATURES
        }
        result = ExtractionResult(features=features, raw_text=raw)
        # model_validator fills any missing features with defaults
        _log_run(variant, user_message, raw, "valid")
        return result
    except (GroqError, ParseError) as exc:
        _log_run(variant, user_message, str(exc), f"fallback:all-assumed ({type(exc).__name__})")
        logger.warning("Feature extraction failed, using all defaults: %s", exc)
        return ExtractionResult(features={})   # model_validator fills all from defaults
    except Exception as exc:
        _log_run(variant, user_message, str(exc), f"fallback:all-assumed (unexpected: {type(exc).__name__})")
        logger.error("Unexpected extraction error: %s", exc)
        return ExtractionResult(features={})


def _predict(extraction: ExtractionResult, user_message: str) -> PredictionResponse:
    """
    Run model.predict() on extracted features, then call LLM to interpret.
    On any failure at interpretation step, returns a minimal PredictionResponse.
    """
    row = extraction.to_model_row()
    df = pd.DataFrame([row])[LOCKED_FEATURES]   # enforce column order
    log_price = _model.predict(df)[0]
    predicted_price = float(np.expm1(log_price))
    price_delta = predicted_price - _MEDIAN_PRICE
    assumed = extraction.assumed_features()

    context_message = (
        f"User description: {user_message}\n"
        f"Extracted features: {json.dumps(row)}\n"
        f"Assumed features: {assumed}\n"
        f"Predicted price: ${predicted_price:,.0f}\n"
        f"Dataset median price: ${_MEDIAN_PRICE:,.0f}"
    )

    try:
        prompt = load_prompt("prediction")
        raw = call_groq(prompt, context_message, json_mode=True)
        data = json.loads(raw)
        result = PredictionResponse.model_validate(
            {
                "verdict": data.get("verdict", "fair"),
                "predicted_price": predicted_price,
                "price_delta": price_delta,
                "assumed_features": assumed,
                "explanation": data.get("explanation", ""),
            }
        )
        _log_run("prediction-v1", user_message, raw, "valid")
        return result
    except Exception as exc:
        _log_run("prediction-v1", user_message, str(exc), f"fallback:minimal ({type(exc).__name__})")
        logger.warning("Prediction interpretation failed, using minimal response: %s", exc)
        return PredictionResponse(
            verdict="fair",
            predicted_price=predicted_price,
            price_delta=price_delta,
            assumed_features=assumed,
            explanation="Prediction complete. Interpretation unavailable.",
        )


def _insight(user_message: str) -> InsightResponse:
    """
    Narrate stats.json to answer a market question.
    On any failure, returns a fallback InsightResponse.
    """
    stats_summary = json.dumps({
        "neighborhood_price_ranges": _stats.get("neighborhood_price_ranges", {}),
        "overall_price_stats": _stats.get("overall_price_stats", {}),
        "feature_distributions": _stats.get("feature_distributions", {}),
        "price_feature_correlations": _stats.get("price_feature_correlations", {}),
    })
    context_message = f"User question: {user_message}\n\nAvailable stats:\n{stats_summary}"

    try:
        prompt = load_prompt("insight")
        raw = call_groq(prompt, context_message, json_mode=True)
        data = json.loads(raw)
        result = InsightResponse.model_validate(data)
        _log_run("insight-v1", user_message, raw, "valid")
        return result
    except Exception as exc:
        _log_run("insight-v1", user_message, str(exc), f"fallback ({type(exc).__name__})")
        logger.warning("Insight failed, using fallback: %s", exc)
        return InsightResponse(
            answer="Unable to retrieve market insights at this time.",
            sources=[],
        )


# --- Public API ---

def run_prediction_chain(user_message: str, variant: str = "extraction-v1") -> PredictionResponse:
    """Full prediction path: extract features -> run model -> interpret."""
    extraction = _extract_features(user_message, variant=variant)
    return _predict(extraction, user_message)


def run_insight_chain(user_message: str) -> InsightResponse:
    """Full insight path: narrate stats.json."""
    return _insight(user_message)


def run_chain(user_message: str, assumed_overrides: dict | None = None) -> Union[PredictionResponse, InsightResponse]:
    """
    Entry point for the full chain.
    Classifies intent, then routes to prediction or insight.
    If assumed_overrides is provided and intent is prediction, overrides are applied
    to the extraction result before prediction runs.
    Never raises — all failures produce a typed fallback response.
    """
    intent = _classify_intent(user_message)
    if intent == "insight":
        return run_insight_chain(user_message)
    # For prediction: extract features, apply overrides if any, then predict
    extraction = _extract_features(user_message)
    if assumed_overrides and intent == "prediction":
        for feat, val in assumed_overrides.items():
            if feat in LOCKED_FEATURES:
                extraction.features[feat] = FeatureFlag(value=val, flag="EXTRACTED")
    return _predict(extraction, user_message)
