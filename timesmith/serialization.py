"""Model serialization and persistence utilities.

This module provides functionality to save and load fitted TimeSmith models.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any, Union

from timesmith.exceptions import ConfigurationError, DataError

logger = logging.getLogger(__name__)

# Try to import joblib for better serialization (optional)
try:
    import joblib

    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


def save_model(
    estimator: Any,
    filepath: Union[str, Path],
    format: str = "auto",
    include_metadata: bool = True,
    **kwargs: Any,
) -> None:
    """Save a fitted estimator to disk.

    Args:
        estimator: Fitted estimator to save (must have is_fitted=True).
        filepath: Path where to save the model.
        format: Serialization format ("auto", "pickle", "joblib").
                "auto" uses joblib if available, otherwise pickle.
        include_metadata: If True, saves metadata (version, class name) alongside model.
        **kwargs: Additional arguments passed to serialization function.

    Raises:
        NotFittedError: If estimator is not fitted.
        ConfigurationError: If format is invalid or save fails.
    """
    from timesmith.exceptions import NotFittedError

    if not hasattr(estimator, "is_fitted") or not estimator.is_fitted:
        raise NotFittedError(estimator.__class__.__name__)

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Determine format
    if format == "auto":
        format = "joblib" if HAS_JOBLIB else "pickle"

    # Save model
    try:
        if format == "joblib":
            if not HAS_JOBLIB:
                raise ConfigurationError(
                    "joblib format requested but joblib is not installed. "
                    "Install with: pip install joblib"
                )
            joblib.dump(estimator, filepath, **kwargs)
        elif format == "pickle":
            with open(filepath, "wb") as f:
                pickle.dump(estimator, f, **kwargs)
        else:
            raise ConfigurationError(
                f"Unknown format: {format}. Must be 'auto', 'pickle', or 'joblib'"
            )

        logger.info(f"Saved {estimator.__class__.__name__} to {filepath}")

        # Save metadata if requested
        if include_metadata:
            metadata_path = filepath.with_suffix(filepath.suffix + ".metadata.json")
            metadata = {
                "class_name": estimator.__class__.__name__,
                "module": estimator.__class__.__module__,
                "is_fitted": estimator.is_fitted,
                "format": format,
            }
            # Add version if available
            try:
                import timesmith

                metadata["timesmith_version"] = timesmith.__version__
            except (ImportError, AttributeError):
                pass

            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

    except Exception as e:
        raise ConfigurationError(
            f"Failed to save model to {filepath}: {e}",
            context={"filepath": str(filepath), "format": format},
        ) from e


def load_model(
    filepath: Union[str, Path],
    format: str = "auto",
    check_fitted: bool = True,
) -> Any:
    """Load a fitted estimator from disk.

    Args:
        filepath: Path to the saved model file.
        format: Serialization format ("auto", "pickle", "joblib").
                "auto" tries to detect from file extension or uses joblib if available.
        check_fitted: If True, verifies that loaded model is fitted.

    Returns:
        Loaded estimator instance.

    Raises:
        DataError: If file doesn't exist or is corrupted.
        ConfigurationError: If format detection fails or model is invalid.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise DataError(f"Model file not found: {filepath}")

    # Determine format
    if format == "auto":
        if filepath.suffix == ".pkl" or filepath.suffix == ".pickle":
            format = "pickle"
        elif filepath.suffix == ".joblib" or filepath.suffix == ".jl":
            format = "joblib"
        else:
            # Default to joblib if available, otherwise pickle
            format = "joblib" if HAS_JOBLIB else "pickle"

    # Load model
    try:
        if format == "joblib":
            if not HAS_JOBLIB:
                raise ConfigurationError(
                    "joblib format requested but joblib is not installed. "
                    "Install with: pip install joblib"
                )
            estimator = joblib.load(filepath)
        elif format == "pickle":
            with open(filepath, "rb") as f:
                estimator = pickle.load(f)
        else:
            raise ConfigurationError(
                f"Unknown format: {format}. Must be 'auto', 'pickle', or 'joblib'"
            )

        # Check if fitted
        if check_fitted:
            if not hasattr(estimator, "is_fitted") or not estimator.is_fitted:
                logger.warning(
                    f"Loaded model {estimator.__class__.__name__} is not fitted. "
                    "Call fit() before using."
                )

        logger.info(f"Loaded {estimator.__class__.__name__} from {filepath}")
        return estimator

    except Exception as e:
        raise DataError(
            f"Failed to load model from {filepath}: {e}",
            context={"filepath": str(filepath), "format": format},
        ) from e


def get_model_metadata(filepath: Union[str, Path]) -> dict:
    """Get metadata for a saved model without loading it.

    Args:
        filepath: Path to the saved model file.

    Returns:
        Dictionary with model metadata.

    Raises:
        DataError: If metadata file doesn't exist.
    """
    filepath = Path(filepath)
    metadata_path = filepath.with_suffix(filepath.suffix + ".metadata.json")

    if not metadata_path.exists():
        raise DataError(f"Metadata file not found: {metadata_path}")

    with open(metadata_path, "r") as f:
        return json.load(f)
