"""Tests for food domain models."""

import pytest
from pathlib import Path

from biotech_ml.base import ModelNotLoadedError


class TestNutriScorePredictor:
    def test_score_to_grade_a(self):
        from biotech_ml.food.nutriscore_predictor import _score_to_grade
        assert _score_to_grade(-5) == "A"

    def test_score_to_grade_e(self):
        from biotech_ml.food.nutriscore_predictor import _score_to_grade
        assert _score_to_grade(25) == "E"

    def test_compute_nutriscore_zero_nutrients(self):
        from biotech_ml.food.nutriscore_predictor import _compute_nutriscore_points
        result = _compute_nutriscore_points({})
        assert result["negative_points"] == 0
        assert result["positive_points"] == 0
        assert result["final_score"] == 0

    def test_nutriscore_rule_fallback(self, tmp_path):
        from biotech_ml.food.nutriscore_predictor import NutriScorePredictor
        model = NutriScorePredictor()
        model.load(tmp_path)
        result = model.predict({
            "energy_kcal": 50,
            "fat": 1,
            "saturated_fat": 0.5,
            "sugar": 2,
            "salt": 0.1,
            "protein": 10,
            "fiber": 5,
        })
        assert result["grade"] in ("A", "B", "C", "D", "E")

    def test_predict_before_load_raises(self):
        from biotech_ml.food.nutriscore_predictor import NutriScorePredictor
        model = NutriScorePredictor()
        with pytest.raises(ModelNotLoadedError):
            model.predict({"energy_kcal": 100})


class TestAllergenNER:
    def test_detect_milk_allergen(self, tmp_path):
        from biotech_ml.food.allergen_ner import AllergenNER
        model = AllergenNER()
        model.load(tmp_path)

        result = model.predict({"ingredient_text": "Contains milk and egg whites"})
        categories = {a["category"] for a in result["allergens"]}
        assert "Milk" in categories
        assert "Eggs" in categories

    def test_empty_text(self, tmp_path):
        from biotech_ml.food.allergen_ner import AllergenNER
        model = AllergenNER()
        model.load(tmp_path)
        result = model.predict({"ingredient_text": ""})
        assert result["allergens"] == []


class TestAdditiveRiskScorer:
    def test_known_additive_safe(self, tmp_path):
        from biotech_ml.food.additive_risk import AdditiveRiskScorer
        model = AdditiveRiskScorer()
        model.load(tmp_path)
        result = model.predict({"additives": ["E330"]})
        assert result["risk_score"] == 0.0

    def test_known_additive_high(self, tmp_path):
        from biotech_ml.food.additive_risk import AdditiveRiskScorer
        model = AdditiveRiskScorer()
        model.load(tmp_path)
        result = model.predict({"additives": ["E171"]})
        assert result["risk_score"] > 0.5

    def test_empty_additives(self, tmp_path):
        from biotech_ml.food.additive_risk import AdditiveRiskScorer
        model = AdditiveRiskScorer()
        model.load(tmp_path)
        result = model.predict({"additives": []})
        assert result["risk_score"] == 0.0


class TestHACCPClassifier:
    def test_biological_hazard_keywords(self):
        from biotech_ml.food.haccp_classifier import _classify_by_keywords
        hazard_detected, category, score = _classify_by_keywords("Salmonella detected in raw chicken")
        assert hazard_detected
        assert category == "biological"

    def test_no_hazard_text_keywords(self):
        from biotech_ml.food.haccp_classifier import _classify_by_keywords
        hazard_detected, category, score = _classify_by_keywords("All systems operating normally")
        assert not hazard_detected
        assert category == "no_hazard_detected"

    def test_haccp_fallback_via_load(self, tmp_path):
        from biotech_ml.food.haccp_classifier import HACCPClassifier
        model = HACCPClassifier()
        model.load(tmp_path)
        result = model.predict({"text": "Salmonella contamination found"})
        assert result["hazard_detected"]
        assert result["category"] == "biological"


class TestIngredientNER:
    def test_parse_simple_ingredients(self, tmp_path):
        from biotech_ml.food.ingredient_ner import IngredientNER
        model = IngredientNER()
        model.load(tmp_path)
        result = model.predict({"text": "sugar, salt, flour"})
        names = [i["name"] for i in result["ingredients"]]
        assert "sugar" in names
        assert "salt" in names
        assert "flour" in names

    def test_parse_with_quantity(self):
        from biotech_ml.food.ingredient_ner import _parse_single_ingredient
        result = _parse_single_ingredient("100g sugar")
        assert result["quantity"] == 100.0
        assert result["unit"] == "g"
