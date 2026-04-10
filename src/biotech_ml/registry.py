import logging
from pathlib import Path

from biotech_ml.base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self, domain: str = "all", model_dir: str = "./artifacts") -> None:
        self._domain = domain.lower()
        self._model_dir = Path(model_dir)
        self._models: dict[str, BaseModel] = {}

    def register(self, model_id: str, model: BaseModel) -> None:
        if model.model_id != model_id:
            raise ValueError(
                f"model_id mismatch: register() called with '{model_id}' "
                f"but model.model_id is '{model.model_id}'"
            )
        if not self._should_load(model_id):
            logger.debug("Skipping model %s (domain=%s)", model_id, self._domain)
            return
        self._models[model_id] = model
        logger.info("Registered model: %s", model_id)

    def get(self, model_id: str) -> BaseModel:
        model = self._models.get(model_id)
        if model is None:
            raise KeyError(f"Model '{model_id}' not found in registry")

        if not model.is_loaded:
            artifact_path = self._model_dir / model_id
            logger.info("Lazy-loading model: %s from %s", model_id, artifact_path)
            model.load(artifact_path)

        return model

    def list_models(self) -> list[str]:
        return list(self._models.keys())

    def health_check(self) -> dict:
        loaded = [model_id for model_id, model in self._models.items() if model.is_loaded]
        return {
            "registered_count": len(self._models),
            "loaded_count": len(loaded),
            "loaded_models": loaded,
        }

    def _should_load(self, model_id: str) -> bool:
        if self._domain == "all":
            return True
        prefix = model_id.split(".")[0] if "." in model_id else model_id.split("_")[0]
        return prefix == self._domain
