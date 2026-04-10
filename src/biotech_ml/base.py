from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from biotech_ml.exceptions import ModelNotLoadedError


class BaseModel(ABC):

    def __init__(self) -> None:
        self._loaded: bool = False

    @property
    @abstractmethod
    def model_id(self) -> str: ...

    @abstractmethod
    def load(self, artifact_path: Path) -> None: ...

    @abstractmethod
    def predict(self, input_data: dict[str, Any]) -> dict[str, Any]: ...

    @abstractmethod
    def metadata(self) -> dict[str, Any]: ...

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise ModelNotLoadedError(
                f"{self.model_id}: load() must be called before predict()"
            )
