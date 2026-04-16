"""Allergen NER using spaCy EntityRuler / NER pipeline."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)

ALLERGEN_PATTERNS: dict[str, list[str]] = {
    "Milk": [
        "milk", "lactose", "casein", "caseinate", "whey", "cream", "butter",
        "cheese", "yogurt", "yoghurt", "ghee", "lactalbumin", "lactoferrin",
        "curds", "custard",
    ],
    "Eggs": [
        "egg", "eggs", "albumin", "globulin", "lysozyme", "mayonnaise",
        "meringue", "ovalbumin", "ovomucin", "ovomucoid", "ovovitellin",
    ],
    "Fish": [
        "fish", "cod", "salmon", "tuna", "anchovy", "anchovies", "bass",
        "catfish", "flounder", "haddock", "halibut", "herring", "mackerel",
        "perch", "pike", "pollock", "snapper", "sole", "swordfish", "tilapia",
        "trout", "fish sauce", "fish oil",
    ],
    "Shellfish": [
        "shellfish", "shrimp", "crab", "lobster", "crayfish", "crawfish",
        "prawn", "scallop", "clam", "mussel", "oyster", "squid", "calamari",
        "octopus", "snail", "abalone",
    ],
    "Tree Nuts": [
        "almond", "almonds", "cashew", "cashews", "walnut", "walnuts",
        "pecan", "pecans", "pistachio", "pistachios", "macadamia",
        "brazil nut", "brazil nuts", "hazelnut", "hazelnuts", "filbert",
        "chestnut", "chestnuts", "pine nut", "pine nuts", "praline",
        "marzipan", "nougat",
    ],
    "Peanuts": [
        "peanut", "peanuts", "groundnut", "groundnuts", "arachis",
        "monkey nut", "monkey nuts", "beer nut", "beer nuts",
    ],
    "Wheat": [
        "wheat", "flour", "bread", "breadcrumb", "breadcrumbs", "bulgur",
        "couscous", "durum", "einkorn", "emmer", "farina", "kamut",
        "semolina", "spelt", "triticale", "gluten",
    ],
    "Soybeans": [
        "soy", "soya", "soybean", "soybeans", "edamame", "miso", "natto",
        "tempeh", "tofu", "soy sauce", "soy lecithin", "soy protein",
    ],
    "Sesame": [
        "sesame", "tahini", "halvah", "halva", "hummus", "sesame oil",
        "sesame seed", "sesame seeds",
    ],
}


def _build_compiled_patterns() -> list[tuple[re.Pattern[str], str]]:
    compiled: list[tuple[re.Pattern[str], str]] = []
    for category, terms in ALLERGEN_PATTERNS.items():
        sorted_terms = sorted(terms, key=len, reverse=True)
        for term in sorted_terms:
            pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
            compiled.append((pattern, category))
    return compiled


class AllergenNER(BaseModel):
    """Detects Big 9 allergens in ingredient text."""

    def __init__(self) -> None:
        super().__init__()
        self._nlp: Any = None
        self._compiled_patterns: list[tuple[re.Pattern[str], str]] = []
        self._use_spacy: bool = False
        self._version: str = "0.1.0"

    @property
    def model_id(self) -> str:
        return "food.allergen_ner"

    def load(self, artifact_path: Path) -> None:
        spacy_model_path = artifact_path / "allergen_ner_model"
        if spacy_model_path.exists():
            try:
                import spacy
                self._nlp = spacy.load(spacy_model_path)
                self._use_spacy = True
                logger.info("Loaded spaCy allergen NER model from %s", spacy_model_path)
            except Exception as exc:
                logger.warning("Failed to load spaCy model: %s, using rule-based", exc)
                self._use_spacy = False
        else:
            logger.info("No spaCy model at %s, using rule-based allergen detection", spacy_model_path)
            self._use_spacy = False

        self._compiled_patterns = _build_compiled_patterns()
        self._loaded = True

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        text: str = input_data.get("ingredient_text", "")
        if not text:
            return {"allergens": []}

        if self._use_spacy and self._nlp is not None:
            return self._predict_spacy(text)
        return self._predict_rules(text)

    def _predict_spacy(self, text: str) -> dict[str, Any]:
        doc = self._nlp(text)
        allergens = []
        for entity in doc.ents:
            if entity.label_ in ALLERGEN_PATTERNS:
                allergens.append({
                    "name": entity.text,
                    "match_type": "spacy_ner",
                    "span_start": entity.start_char,
                    "span_end": entity.end_char,
                    "category": entity.label_,
                })
        return {"allergens": allergens}

    def _predict_rules(self, text: str) -> dict[str, Any]:
        allergens: list[dict[str, Any]] = []
        seen_spans: set[tuple[int, int]] = set()

        for pattern, category in self._compiled_patterns:
            for match in pattern.finditer(text):
                span = (match.start(), match.end())
                if any(existing[0] <= span[0] < existing[1] or existing[0] < span[1] <= existing[1] for existing in seen_spans):
                    continue
                seen_spans.add(span)
                allergens.append({
                    "name": match.group(),
                    "match_type": "rule_based",
                    "span_start": span[0],
                    "span_end": span[1],
                    "category": category,
                })

        allergens.sort(key=lambda a: a["span_start"])
        return {"allergens": allergens}

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "description": "Big 9 allergen detection (spaCy NER / rule-based fallback)",
            "categories": list(ALLERGEN_PATTERNS.keys()),
            "backend": "spacy" if self._use_spacy else "rule_based",
        }
