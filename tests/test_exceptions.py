"""Tests for TimeSmith custom exceptions."""

import pytest

from timesmith.exceptions import (
    ConfigurationError,
    DataError,
    ForecastError,
    NotFittedError,
    PipelineError,
    TimeSmithError,
    TransformError,
    UnsupportedOperationError,
    ValidationError,
)


class TestTimeSmithError:
    """Tests for base TimeSmithError."""

    def test_basic_error(self):
        """Test basic error creation."""
        error = TimeSmithError("Test error")
        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.context == {}

    def test_error_with_context(self):
        """Test error with context."""
        error = TimeSmithError("Test error", context={"key": "value", "num": 42})
        assert "Test error" in str(error)
        assert "key=value" in str(error)
        assert "num=42" in str(error)
        assert error.context == {"key": "value", "num": 42}

    def test_error_inheritance(self):
        """Test that all errors inherit from TimeSmithError."""
        errors = [
            ValidationError("test"),
            DataError("test"),
            ForecastError("test"),
            TransformError("test"),
            PipelineError("test"),
            ConfigurationError("test"),
        ]
        for error in errors:
            assert isinstance(error, TimeSmithError)


class TestNotFittedError:
    """Tests for NotFittedError."""

    def test_not_fitted_error(self):
        """Test NotFittedError creation."""
        error = NotFittedError("TestEstimator")
        assert "TestEstimator" in str(error)
        assert "not fitted" in str(error)
        assert "fit" in str(error)
        assert error.context["estimator"] == "TestEstimator"


class TestUnsupportedOperationError:
    """Tests for UnsupportedOperationError."""

    def test_unsupported_operation_error(self):
        """Test UnsupportedOperationError creation."""
        error = UnsupportedOperationError("predict_interval", "TestForecaster")
        assert "TestForecaster" in str(error)
        assert "predict_interval" in str(error)
        assert error.context["operation"] == "predict_interval"
        assert error.context["estimator"] == "TestForecaster"

    def test_unsupported_operation_with_reason(self):
        """Test UnsupportedOperationError with reason."""
        error = UnsupportedOperationError(
            "predict_interval", "TestForecaster", reason="Not implemented yet"
        )
        assert "Not implemented yet" in str(error)


class TestSpecificErrors:
    """Tests for specific error types."""

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid input", context={"input_type": "str"})
        assert isinstance(error, TimeSmithError)
        assert error.message == "Invalid input"

    def test_data_error(self):
        """Test DataError."""
        error = DataError("Insufficient data", context={"n_samples": 5, "required": 10})
        assert isinstance(error, TimeSmithError)
        assert error.context["n_samples"] == 5

    def test_forecast_error(self):
        """Test ForecastError."""
        error = ForecastError("Forecast failed")
        assert isinstance(error, TimeSmithError)

    def test_transform_error(self):
        """Test TransformError."""
        error = TransformError("Transform failed")
        assert isinstance(error, TimeSmithError)

    def test_pipeline_error(self):
        """Test PipelineError."""
        error = PipelineError("Pipeline step failed", context={"step": 2})
        assert isinstance(error, TimeSmithError)
        assert error.context["step"] == 2

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid parameter", context={"param": "alpha"})
        assert isinstance(error, TimeSmithError)
        assert error.context["param"] == "alpha"

