# app/schemas.py
# Pydantic models for all chain inputs and outputs

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator

ROOT_DIR = Path(__file__).resolve().parent.parent
_defaults_path = ROOT_DIR / "data" / "defaults.json"
_defaults_raw = json.loads(_defaults_path.read_text())
LOCKED_FEATURES: list[str] = _defaults_raw["locked_features"]
FEATURE_DEFAULTS: dict = _defaults_raw["defaults"]


# --- Intent ---

class IntentResult(BaseModel):
    intent: Literal["prediction", "insight"]


# --- Extraction ---

class FeatureFlag(BaseModel):
    value: float | str
    flag: Literal["EXTRACTED", "ASSUMED"] = "ASSUMED"


class ExtractionResult(BaseModel):
    features: dict[str, FeatureFlag] = Field(default_factory=dict)
    raw_text: str = ""

    @model_validator(mode="after")
    def fill_missing_with_defaults(self) -> "ExtractionResult":
        for feat in LOCKED_FEATURES:
            if feat not in self.features:
                self.features[feat] = FeatureFlag(
                    value=FEATURE_DEFAULTS[feat], flag="ASSUMED"
                )
        return self

    def to_model_row(self) -> dict:
        """Return {feature: value} dict for model.predict()."""
        return {k: v.value for k, v in self.features.items()}

    def assumed_features(self) -> list[str]:
        return [k for k, v in self.features.items() if v.flag == "ASSUMED"]


# --- Prediction ---

class PredictionResponse(BaseModel):
    verdict: Literal["fair", "high", "low"]
    predicted_price: float
    price_delta: float          # positive = above median, negative = below
    assumed_features: list[str] = Field(default_factory=list)
    explanation: str = ""


# --- Insight ---

class InsightResponse(BaseModel):
    answer: str
    sources: list[str] = Field(default_factory=list)   # keys from stats.json used


# --- Error types (not Pydantic — plain exceptions) ---

class GroqError(Exception):
    """Raised when Groq API call fails."""

class ParseError(Exception):
    """Raised when LLM output cannot be parsed as JSON."""
