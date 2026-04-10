"""Medical / clinical diagnostics domain models."""

__all__ = [
    "ResultAnomalyDetector",
    "DeltaChecker",
    "DrugLabInteraction",
    "ClinicalQA",
    "DDxSuggester",
    "TerminologyMapper",
]

_MODEL_MAP = {
    "ResultAnomalyDetector": "biotech_ml.medical.anomaly_detector",
    "DeltaChecker": "biotech_ml.medical.delta_check",
    "DrugLabInteraction": "biotech_ml.medical.drug_lab_interaction",
    "ClinicalQA": "biotech_ml.medical.clinical_qa",
    "DDxSuggester": "biotech_ml.medical.ddx_suggester",
    "TerminologyMapper": "biotech_ml.medical.terminology_mapper",
}


def __getattr__(name: str):
    """Lazy imports to avoid pulling in all sub-modules at import time."""
    if name in _MODEL_MAP:
        import importlib
        mod = importlib.import_module(_MODEL_MAP[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
