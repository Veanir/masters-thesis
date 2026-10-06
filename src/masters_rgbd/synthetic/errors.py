"""Project-specific exceptions."""


class DatasetError(RuntimeError):
    """Base class for dataset generation errors."""


class DependencyMissingError(DatasetError):
    """Raised when an optional runtime dependency is not installed."""


class AssetError(DatasetError):
    """Raised when source or prepared object assets are invalid."""


class GenerationError(DatasetError):
    """Raised when a scene cannot be generated."""


class ValidationError(DatasetError):
    """Raised when generated dataset contents are invalid."""
