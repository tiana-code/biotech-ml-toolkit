"""Food safety domain models - nutriscore, allergen NER, additive risk, HACCP, ingredient NER, nutritional anomaly, product lookup."""

__all__ = [
    "NutriScorePredictor",
    "AllergenNER",
    "AdditiveRiskScorer",
    "HACCPClassifier",
    "IngredientNER",
    "NutritionalAnomalyDetector",
    "ProductLookup",
]

_MODEL_MAP = {
    "NutriScorePredictor": "biotech_ml.food.nutriscore_predictor",
    "AllergenNER": "biotech_ml.food.allergen_ner",
    "AdditiveRiskScorer": "biotech_ml.food.additive_risk",
    "HACCPClassifier": "biotech_ml.food.haccp_classifier",
    "IngredientNER": "biotech_ml.food.ingredient_ner",
    "NutritionalAnomalyDetector": "biotech_ml.food.nutritional_anomaly",
    "ProductLookup": "biotech_ml.food.product_lookup",
}


def __getattr__(name: str):
    """Lazy imports to avoid pulling in optional deps at import time."""
    if name in _MODEL_MAP:
        import importlib
        mod = importlib.import_module(_MODEL_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
