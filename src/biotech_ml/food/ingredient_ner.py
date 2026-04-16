"""Ingredient NER - parses raw ingredient text into structured data."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

_QUANTITY_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mg|g|kg|ml|l|oz|lb|tbsp|tsp|cup|cups|%)\b",
    re.IGNORECASE,
)

_PERCENTAGE_RE = re.compile(
    r"\(?\s*(\d+(?:[.,]\d+)?)\s*%\s*\)?",
)

_SEPARATOR_RE = re.compile(r"[,;]\s*|\band\b\s*", re.IGNORECASE)

_PAREN_RE = re.compile(r"\(([^)]*)\)")

_UNIT_ALIASES: dict[str, str] = {
    "mg": "mg", "g": "g", "kg": "kg",
    "ml": "ml", "l": "l",
    "oz": "oz", "lb": "lb",
    "tbsp": "tbsp", "tsp": "tsp",
    "cup": "cup", "cups": "cup",
}


def _parse_single_ingredient(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if not raw:
        return {"name": "", "quantity": None, "unit": None, "percentage": None}

    percentage: float | None = None
    quantity: float | None = None
    unit: str | None = None

    pct_match = _PERCENTAGE_RE.search(raw)
    if pct_match:
        try:
            percentage = float(pct_match.group(1).replace(",", "."))
        except ValueError:
            pass

    qu_match = _QUANTITY_UNIT_RE.search(raw)
    if qu_match:
        try:
            quantity = float(qu_match.group(1).replace(",", "."))
        except ValueError:
            pass
        raw_unit = qu_match.group(2).lower()
        if raw_unit == "%":
            if percentage is None:
                percentage = quantity
            quantity = None
        else:
            unit = _UNIT_ALIASES.get(raw_unit, raw_unit)

    name = raw
    name = _QUANTITY_UNIT_RE.sub("", name)
    name = _PERCENTAGE_RE.sub("", name)
    name = _PAREN_RE.sub("", name)
    name = re.sub(r"\s+", " ", name).strip(" ,;.-")

    if not name:
        name = raw.strip()

    return {
        "name": name,
        "quantity": quantity,
        "unit": unit,
        "percentage": percentage,
    }


class IngredientNER(BaseModel):
    """Ingredient text parser with optional spaCy NER backend."""

    def __init__(self) -> None:
        super().__init__()
        self._nlp: Any = None
        self._use_spacy: bool = False
        self._version: str = "0.1.0"

    @property
    def model_id(self) -> str:
        return "food.ingredient_ner"

    def load(self, artifact_path: Path) -> None:
        spacy_path = artifact_path / "ingredient_ner_model"
        if spacy_path.exists():
            try:
                import spacy
                self._nlp = spacy.load(spacy_path)
                self._use_spacy = True
                logger.info("Loaded spaCy ingredient NER from %s", spacy_path)
            except Exception as exc:
                logger.warning("Failed to load spaCy model: %s, using rules", exc)
                self._use_spacy = False
        else:
            logger.info("No spaCy model at %s, using rule-based ingredient parser", spacy_path)
            self._use_spacy = False

        self._loaded = True

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        text: str = input_data.get("text", "")
        if not text.strip():
            return {"ingredients": []}

        if self._use_spacy and self._nlp is not None:
            return self._predict_spacy(text)
        return self._predict_rules(text)

    def _predict_spacy(self, text: str) -> dict[str, Any]:
        doc = self._nlp(text)
        ingredients = []
        for entity in doc.ents:
            parsed = _parse_single_ingredient(entity.text)
            if parsed["name"]:
                ingredients.append(parsed)
        if not ingredients:
            return self._predict_rules(text)
        return {"ingredients": ingredients}

    def _predict_rules(self, text: str) -> dict[str, Any]:
        parts = _SEPARATOR_RE.split(text)
        ingredients = []
        for part in parts:
            part = part.strip()
            if not part:
                continue
            parsed = _parse_single_ingredient(part)
            if parsed["name"]:
                ingredients.append(parsed)
        return {"ingredients": ingredients}

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "description": "Ingredient text parser (spaCy NER / rule-based fallback)",
            "backend": "spacy" if self._use_spacy else "rule_based",
        }
