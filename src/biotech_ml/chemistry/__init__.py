"""Chemistry domain models - solubility, lipophilicity, toxicity, GHS, allergen detection, INCI safety."""

__all__ = [
    "ToxicityScorer",
    "SolubilityPredictor",
    "LipophilicityPredictor",
    "CosmeticAllergenDetector",
    "GHSClassifier",
    "INCISafetyScorer",
]

_MODEL_MAP = {
    "ToxicityScorer": "biotech_ml.chemistry.toxicity_scorer",
    "SolubilityPredictor": "biotech_ml.chemistry.solubility_predictor",
    "LipophilicityPredictor": "biotech_ml.chemistry.lipophilicity_predictor",
    "CosmeticAllergenDetector": "biotech_ml.chemistry.allergen_detector",
    "GHSClassifier": "biotech_ml.chemistry.ghs_classifier",
    "INCISafetyScorer": "biotech_ml.chemistry.inci_safety_score",
}


def __getattr__(name: str):
    """Lazy imports to avoid pulling in rdkit at import time."""
    if name in _MODEL_MAP:
        import importlib
        mod = importlib.import_module(_MODEL_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
