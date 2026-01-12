"""Ensemble forecasting using classification + regression approach.

This forecaster uses a hybrid approach where:
- Classification model predicts direction (increase/decrease)
- Regression model predicts magnitude using classification predictions as features

This approach can outperform standard ARIMA models for complex patterns and noisy data.
"""

import logging
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import numpy as np
import pandas as pd

from timesmith.core.base import BaseForecaster
from timesmith.core.tags import set_tags
from timesmith.results.forecast import Forecast

if TYPE_CHECKING:
    from timesmith.typing import SeriesLike, TableLike

logger = logging.getLogger(__name__)

# Optional sklearn for ensemble methods
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning(
        "scikit-learn not available. EnsembleForecaster requires sklearn. "
        "Install with: pip install scikit-learn"
    )


class EnsembleForecaster(BaseForecaster):
    """Ensemble forecaster combining classification and regression models.

    Uses Random Forest classifier to predict direction (up/down) and Random Forest
    regressor to predict magnitude, with classification predictions as features.
    """

    def __init__(
        self,
        n_lags: int = 2,
        random_state: Optional[int] = None,
        n_estimators_classifier: int = 100,
        n_estimators_regressor: int = 100,
        max_depth: Optional[int] = None,
    ):
        """Initialize ensemble forecaster.

        Args:
            n_lags: Number of lagged features to use.
            random_state: Random seed for reproducibility.
            n_estimators_classifier: Number of trees for classifier.
            n_estimators_regressor: Number of trees for regressor.
            max_depth: Maximum depth of trees (None = unlimited).
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError(
                "scikit-learn is required for EnsembleForecaster. "
                "Install with: pip install scikit-learn"
            )

        super().__init__()
        self.n_lags = n_lags
        self.random_state = random_state

        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators_classifier,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

        self.regressor = RandomForestRegressor(
            n_estimators=n_estimators_regressor,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )

        self.scaler = StandardScaler()

        set_tags(
            self,
            scitype_input="SeriesLike",
            scitype_output="ForecastLike",
            handles_missing=False,
            requires_sorted_index=True,
        )

    def fit(
        self,
        y: Union["SeriesLike", Any],
        X: Optional[Union["TableLike", Any]] = None,
        **fit_params: Any,
    ) -> "EnsembleForecaster":
        """Fit ensemble models on training data.

        Args:
            y: Target time series.
            X: Optional exogenous data (not yet supported).
            **fit_params: Additional fit parameters.

        Returns:
            Self for method chaining.
        """
        if X is not None:
            logger.warning("Exogenous data X not yet supported in EnsembleForecaster")

        if isinstance(y, pd.Series):
            self.y_ = y.values
            self.index_ = y.index
        elif isinstance(y, pd.DataFrame) and y.shape[1] == 1:
            self.y_ = y.iloc[:, 0].values
            self.index_ = y.index
        else:
            self.y_ = np.asarray(y, dtype=float)
            self.index_ = np.arange(len(self.y_))

        # Remove invalid values
        valid_mask = np.isfinite(self.y_)
        self.y_ = self.y_[valid_mask]
        self.index_ = self.index_[valid_mask]

        if len(self.y_) < self.n_lags + 10:
            raise ValueError(
                f"Need at least {self.n_lags + 10} data points for training"
            )

        # Create features
        X_features, y_class, y_reg = self._create_features(self.y_)

        if len(X_features) < 20:
            raise ValueError("Not enough data points after feature creation")

        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)

        # Train classification model
        self.classifier.fit(X_scaled, y_class)

        # Get classification predictions for training
        y_class_pred = self.classifier.predict(X_scaled)

        # Add classification predictions as feature for regression
        X_reg = np.column_stack([X_scaled, y_class_pred])

        # Train regression model
        self.regressor.fit(X_reg, y_reg)

        self._is_fitted = True
        return self

    def predict(
        self,
        fh: Union[int, list, Any],
        X: Optional[Union["TableLike", Any]] = None,
        **predict_params: Any,
    ) -> Forecast:
        """Generate forecasts.

        Args:
            fh: Forecast horizon (integer or array-like).
            X: Optional exogenous data (not yet supported).
            **predict_params: Additional prediction parameters.

        Returns:
            Forecast results.
        """
        self._check_is_fitted()

        if X is not None:
            logger.warning("Exogenous data X not yet supported in EnsembleForecaster")

        # Convert fh to integer
        if isinstance(fh, (int, np.integer)):
            n_steps = int(fh)
        elif isinstance(fh, (list, np.ndarray, pd.Index)):
            n_steps = len(fh)
            fh_arr = np.asarray(fh)
        else:
            raise ValueError(f"Unsupported fh type: {type(fh)}")

        # Recursive prediction
        predictions = []
        current_data = self.y_.copy()

        for step in range(n_steps):
            # Create features from current data
            X_step, _, _ = self._create_features(current_data)

            if len(X_step) == 0:
                # Not enough data, use last value
                predictions.append(current_data[-1])
                continue

            # Use last row
            X_step_last = (
                X_step.iloc[[-1]] if isinstance(X_step, pd.DataFrame) else X_step[-1:]
            )

            # Scale features
            X_step_scaled = self.scaler.transform(X_step_last)

            # Predict direction
            direction_pred = self.classifier.predict(X_step_scaled)

            # Add direction prediction as feature
            X_step_reg = np.column_stack([X_step_scaled, direction_pred])

            # Predict value
            pred_value = self.regressor.predict(X_step_reg)[0]
            predictions.append(pred_value)

            # Append prediction to current_data for next iteration
            current_data = np.append(current_data, pred_value)

        predictions = np.array(predictions)

        # Convert to Series
        if isinstance(fh, (list, np.ndarray, pd.Index)):
            y_pred_series = pd.Series(predictions, index=fh_arr)
        else:
            y_pred_series = pd.Series(predictions)

        return Forecast(
            y_pred=y_pred_series,
            fh=fh,
            metadata={
                "n_lags": self.n_lags,
                "method": "ensemble_classification_regression",
            },
        )

    def _create_features(
        self, data: np.ndarray
    ) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Create features for classification and regression.

        Args:
            data: Time series data.

        Returns:
            Tuple of (features, direction_target, regression_target).
        """
        df = pd.DataFrame({"value": data})

        # Create lagged features
        for lag in range(1, self.n_lags + 1):
            df[f"lag_{lag}"] = df["value"].shift(lag)

        # Rate of change
        df["rate_of_change"] = df["value"].diff()

        # Moving averages
        if len(df) > 5:
            df["ma_3"] = df["value"].rolling(window=3, min_periods=1).mean()
            df["ma_5"] = df["value"].rolling(window=5, min_periods=1).mean()

        # Target for classification: 1 if next value increases, 0 if decreases
        df["direction"] = (df["value"].shift(-1) > df["value"]).astype(int)

        # Target for regression: next value
        df["next_value"] = df["value"].shift(-1)

        # Drop NaN rows
        df = df.dropna()

        if len(df) == 0:
            raise ValueError("Not enough data to create features")

        # Features (excluding targets)
        feature_cols = [
            col for col in df.columns if col not in ["direction", "next_value", "value"]
        ]
        X = df[feature_cols]

        y_class = df["direction"]
        y_reg = df["next_value"]

        return X, y_class, y_reg
