"""Tests for chemistry domain models."""

import pytest
from pathlib import Path

from biotech_ml.base import ModelNotLoadedError


class TestSolubilityPredictor:
    def test_categorize_solubility_highly_soluble(self):
        from biotech_ml.chemistry.solubility_predictor import _categorize_solubility
        assert _categorize_solubility(200.0) == "highly_soluble"

    def test_categorize_solubility_soluble(self):
        from biotech_ml.chemistry.solubility_predictor import _categorize_solubility
        assert _categorize_solubility(10.0) == "soluble"

    def test_categorize_solubility_slightly_soluble(self):
        from biotech_ml.chemistry.solubility_predictor import _categorize_solubility
        assert _categorize_solubility(0.05) == "slightly_soluble"

    def test_categorize_solubility_insoluble(self):
        from biotech_ml.chemistry.solubility_predictor import _categorize_solubility
        assert _categorize_solubility(0.001) == "insoluble"

    def test_log_s_to_mg_l_zero_mw(self):
        from biotech_ml.chemistry.solubility_predictor import _log_s_to_mg_l
        assert _log_s_to_mg_l(-3.0, 0.0) is None

    def test_log_s_to_mg_l_valid(self):
        from biotech_ml.chemistry.solubility_predictor import _log_s_to_mg_l
        result = _log_s_to_mg_l(0.0, 100.0)
        assert result is not None
        assert result == pytest.approx(100000.0)

    def test_predict_before_load_raises(self):
        from biotech_ml.chemistry.solubility_predictor import SolubilityPredictor
        model = SolubilityPredictor()
        with pytest.raises(ModelNotLoadedError):
            model.predict({"smiles": "CCO"})


class TestLipophilicityPredictor:
    def test_skin_penetration_high(self):
        from biotech_ml.chemistry.lipophilicity_predictor import _assess_skin_penetration
        assert _assess_skin_penetration(2.0) == "high"

    def test_skin_penetration_moderate(self):
        from biotech_ml.chemistry.lipophilicity_predictor import _assess_skin_penetration
        assert _assess_skin_penetration(0.5) == "moderate"

    def test_skin_penetration_low(self):
        from biotech_ml.chemistry.lipophilicity_predictor import _assess_skin_penetration
        assert _assess_skin_penetration(-2.0) == "low"

    def test_lipophilicity_categories(self):
        from biotech_ml.chemistry.lipophilicity_predictor import _lipophilicity_category
        assert _lipophilicity_category(-2.0) == "very_hydrophilic"
        assert _lipophilicity_category(0.0) == "hydrophilic"
        assert _lipophilicity_category(2.0) == "moderately_lipophilic"
        assert _lipophilicity_category(4.0) == "lipophilic"
        assert _lipophilicity_category(6.0) == "very_lipophilic"


class TestGHSClassifier:
    def test_signal_word_danger(self):
        from biotech_ml.chemistry.ghs_classifier import GHSClassifier
        assert GHSClassifier._determine_signal_word(["H300", "H315"]) == "Danger"

    def test_signal_word_warning(self):
        from biotech_ml.chemistry.ghs_classifier import GHSClassifier
        assert GHSClassifier._determine_signal_word(["H315", "H319"]) == "Warning"

    def test_signal_word_none(self):
        from biotech_ml.chemistry.ghs_classifier import GHSClassifier
        assert GHSClassifier._determine_signal_word([]) is None

    def test_pictograms_dedup(self):
        from biotech_ml.chemistry.ghs_classifier import GHSClassifier
        pics = GHSClassifier._h_codes_to_pictograms(["H300", "H301"])
        assert pics == ["GHS06"]


class TestAllergenDetector:
    def test_normalize(self):
        from biotech_ml.chemistry.allergen_detector import _normalize
        assert _normalize("  Benzyl Alcohol  ") == "BENZYL ALCOHOL"

    def test_detect_known_allergen(self, tmp_path):
        from biotech_ml.chemistry.allergen_detector import CosmeticAllergenDetector
        detector = CosmeticAllergenDetector()
        detector.load(tmp_path)

        result = detector.predict({"ingredients": ["LINALOOL", "WATER", "GLYCERIN"]})
        assert len(result["allergens"]) == 1
        assert result["allergens"][0]["inci_name"] == "LINALOOL"

    def test_predict_before_load_raises(self):
        from biotech_ml.chemistry.allergen_detector import CosmeticAllergenDetector
        detector = CosmeticAllergenDetector()
        with pytest.raises(ModelNotLoadedError):
            detector.predict({"ingredients": ["LINALOOL"]})


class TestINCISafetyScorer:
    def test_allergen_risk_levels(self):
        from biotech_ml.chemistry.inci_safety_score import _allergen_risk_level
        assert _allergen_risk_level(0) == "none"
        assert _allergen_risk_level(1) == "low"
        assert _allergen_risk_level(3) == "moderate"
        assert _allergen_risk_level(10) == "high"

    def test_allergen_risk_penalty(self):
        from biotech_ml.chemistry.inci_safety_score import _allergen_risk_penalty
        assert _allergen_risk_penalty("none") == 0.0
        assert _allergen_risk_penalty("high") == 3.0

    def test_regulatory_db_exact_match(self):
        from biotech_ml.chemistry.inci_safety_score import REGULATORY_DB_DEFAULT
        assert "FORMALDEHYDE" in REGULATORY_DB_DEFAULT
        assert REGULATORY_DB_DEFAULT["FORMALDEHYDE"]["status"] == "banned"

    def test_regulatory_unknown_returns_not_in_database(self, tmp_path):
        from biotech_ml.chemistry.inci_safety_score import INCISafetyScorer
        scorer = INCISafetyScorer()
        scorer.load(tmp_path)
        result = scorer.predict({"inci_name": "TOTALLY_UNKNOWN_INGREDIENT_XYZ"})
        assert result["status"] == "degraded"
        assert result["details"]["regulatory_status"] == "not_in_database"
