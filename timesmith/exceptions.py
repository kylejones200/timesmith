"""Custom exceptions for TimeSmith.

This module provides a hierarchy of custom exceptions for better error handling
and more informative error messages throughout the TimeSmith library.
"""


class TimeSmithError(Exception):
    """Base exception for all TimeSmith errors.

    All custom exceptions in TimeSmith should inherit from this class.
    This allows users to catch all TimeSmith-specific errors with a single
    exception handler if needed.
    """

    def __init__(self, message: str, context: dict = None):
        """Initialize TimeSmith error.

        Args:
            message: Error message describing what went wrong.
            context: Optional dictionary with additional context about the error.
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __str__(self) -> str:
        """Return formatted error message with context."""
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} (Context: {context_str})"
        return self.message


class ValidationError(TimeSmithError):
    """Raised when input validation fails.

    This exception is raised when data doesn't match expected formats,
    types, or constraints.
    """

    pass


class DataError(TimeSmithError):
    """Raised when there are issues with the data itself.

    Examples: insufficient data points, all NaN values, empty data, etc.
    """

    pass


class ForecastError(TimeSmithError):
    """Raised when forecasting operations fail.

    This includes issues during model fitting, prediction, or interval estimation.
    """

    pass


class TransformError(TimeSmithError):
    """Raised when transformation operations fail.

    This includes issues during fit, transform, or inverse_transform operations.
    """

    pass


class PipelineError(TimeSmithError):
    """Raised when pipeline operations fail.

    This includes issues with pipeline composition, execution, or step failures.
    """

    pass


class ConfigurationError(TimeSmithError):
    """Raised when there are configuration or parameter issues.

    This includes invalid parameter values, missing required parameters, etc.
    """

    pass


class NotFittedError(TimeSmithError):
    """Raised when an operation requires a fitted estimator but it hasn't been fitted.

    This is a more specific error than ValueError for better error handling.
    """

    def __init__(self, estimator_name: str):
        """Initialize NotFittedError.

        Args:
            estimator_name: Name of the estimator that needs to be fitted.
        """
        message = (
            f"This {estimator_name} instance is not fitted yet. "
            "Call 'fit' before using this estimator."
        )
        super().__init__(message, context={"estimator": estimator_name})


class UnsupportedOperationError(TimeSmithError):
    """Raised when an operation is not supported by a particular estimator.

    This is more specific than NotImplementedError for user-facing operations.
    """

    def __init__(self, operation: str, estimator_name: str, reason: str = None):
        """Initialize UnsupportedOperationError.

        Args:
            operation: Name of the unsupported operation.
            estimator_name: Name of the estimator that doesn't support the operation.
            reason: Optional reason why the operation is not supported.
        """
        message = f"{estimator_name} does not support {operation}"
        if reason:
            message += f": {reason}"
        super().__init__(
            message, context={"operation": operation, "estimator": estimator_name}
        )
