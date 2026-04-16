"""Product lookup via DuckDB + OpenFoodFacts Parquet files."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)


class ProductLookup(BaseModel):
    """Looks up product data by barcode using DuckDB over OpenFoodFacts Parquet."""

    def __init__(self) -> None:
        super().__init__()
        self._con: Any = None
        self._parquet_path: Path | None = None
        self._product_count: int = 0
        self._version: str = "0.1.0"

    @property
    def model_id(self) -> str:
        return "food.product_lookup"

    def load(self, artifact_path: Path) -> None:
        parquet_dir = artifact_path / "openfoodfacts"
        parquet_files = list(parquet_dir.glob("*.parquet")) if parquet_dir.exists() else []

        if parquet_files:
            try:
                import duckdb
                self._con = duckdb.connect(database=":memory:")
                self._parquet_path = parquet_dir
                parquet_glob = str(parquet_dir / "*.parquet")
                self._con.execute(
                    f"CREATE VIEW products AS SELECT * FROM read_parquet('{parquet_glob}')"
                )
                count = self._con.execute("SELECT COUNT(*) FROM products").fetchone()
                self._product_count = count[0] if count else 0
                logger.info(
                    "Loaded OpenFoodFacts Parquet (%d products) from %s",
                    self._product_count,
                    parquet_dir,
                )
            except Exception as exc:
                logger.warning("Failed to init DuckDB: %s", exc)
                self._con = None
        else:
            logger.warning("No Parquet files at %s, product lookup will return empty", parquet_dir)
            self._con = None

        self._loaded = True

    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]:
        self._ensure_loaded()
        barcode: str = str(input_data.get("barcode", "")).strip()
        if not barcode:
            return self._empty_response(barcode, "not_found")

        if self._con is None:
            return self._empty_response(barcode, "backend_unavailable")

        try:
            result = self._con.execute(
                """
                SELECT
                    code, product_name, brands, nutriscore_grade,
                    ecoscore_grade, nova_group, allergens, additives_tags,
                    ingredients_text, energy_kcal_100g, fat_100g,
                    saturated_fat_100g, sugars_100g, salt_100g,
                    proteins_100g, fiber_100g, countries_tags, image_url
                FROM products
                WHERE code = ?
                LIMIT 1
                """,
                [barcode],
            ).fetchone()
        except Exception as exc:
            logger.error("DuckDB query failed for barcode %s: %s", barcode, exc)
            return self._empty_response(barcode, "backend_unavailable")

        if result is None:
            return self._empty_response(barcode, "not_found")

        return self._row_to_response(result)

    def _row_to_response(self, row: tuple) -> dict[str, Any]:
        (
            code, product_name, brands, nutriscore_grade, ecoscore_grade,
            nova_group, allergens_raw, additives_raw, ingredients_text,
            energy, fat, sat_fat, sugars, salt, proteins, fiber,
            countries_raw, image_url,
        ) = row

        def _split_tags(raw: Any) -> list[str]:
            if raw is None:
                return []
            if isinstance(raw, list):
                return [str(item) for item in raw]
            return [tag.strip() for tag in str(raw).split(",") if tag.strip()]

        nutrients = {
            "energy_kcal_100g": energy,
            "fat_100g": fat,
            "saturated_fat_100g": sat_fat,
            "sugars_100g": sugars,
            "salt_100g": salt,
            "proteins_100g": proteins,
            "fiber_100g": fiber,
        }
        nutrients = {name: val for name, val in nutrients.items() if val is not None}

        nova = None
        if nova_group is not None:
            try:
                nova = int(nova_group)
                if nova < 1 or nova > 4:
                    nova = None
            except (ValueError, TypeError):
                nova = None

        return {
            "barcode": str(code) if code else "",
            "product_name": product_name,
            "brands": brands,
            "nutriscore_grade": nutriscore_grade,
            "ecoscore_grade": ecoscore_grade,
            "nova_group": nova,
            "allergens": _split_tags(allergens_raw),
            "additives": _split_tags(additives_raw),
            "ingredients_text": ingredients_text,
            "nutrients": nutrients if nutrients else None,
            "countries": _split_tags(countries_raw),
            "image_url": image_url,
        }

    @staticmethod
    def _empty_response(barcode: str, status: str) -> dict[str, Any]:
        return {
            "barcode": barcode,
            "status": status,
            "product_name": None,
            "brands": None,
            "nutriscore_grade": None,
            "ecoscore_grade": None,
            "nova_group": None,
            "allergens": [],
            "additives": [],
            "ingredients_text": None,
            "nutrients": None,
            "countries": [],
            "image_url": None,
        }

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "version": self._version,
            "description": "Product lookup via DuckDB + OpenFoodFacts Parquet",
            "product_count": self._product_count,
            "backend": "duckdb" if self._con is not None else "unavailable",
        }
