"""Custom exception types for biotech-ml-toolkit."""


class ModelNotLoadedError(RuntimeError):
    """Raised when predict() is called before load()."""


class InputValidationError(ValueError):
    """Raised when input data fails validation."""


class ArtifactLoadError(RuntimeError):
    """Raised when model artifacts cannot be loaded."""


class UnsupportedInputError(ValueError):
    """Raised when input contains unsupported features, tests, or codes."""
