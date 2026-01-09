"""Tests for core base classes."""

import pytest

from timesmith.core.base import (
    BaseDetector,
    BaseEstimator,
    BaseFeaturizer,
    BaseForecaster,
    BaseObject,
    BaseTransformer,
)


class TestBaseObject:
    """Tests for BaseObject."""

    def test_get_params(self):
        """Test get_params method."""

        class TestObject(BaseObject):
            def __init__(self, param1=1, param2="test"):
                self.param1 = param1
                self.param2 = param2

        obj = TestObject(param1=10, param2="hello")
        params = obj.get_params()
        assert params["param1"] == 10
        assert params["param2"] == "hello"

    def test_set_params(self):
        """Test set_params method."""

        class TestObject(BaseObject):
            def __init__(self, param1=1):
                self.param1 = param1

        obj = TestObject()
        obj.set_params(param1=20)
        assert obj.param1 == 20

    def test_clone(self):
        """Test clone method."""

        class TestObject(BaseObject):
            def __init__(self, param1=1):
                self.param1 = param1

        obj = TestObject(param1=30)
        cloned = obj.clone()
        assert cloned.param1 == 30
        assert cloned is not obj

    def test_repr(self):
        """Test __repr__ method."""

        class TestObject(BaseObject):
            def __init__(self, param1=1, param2="test"):
                self.param1 = param1
                self.param2 = param2

        obj = TestObject(param1=10, param2="hello")
        repr_str = repr(obj)
        assert "TestObject" in repr_str
        assert "param1=10" in repr_str


class TestBaseEstimator:
    """Tests for BaseEstimator."""

    def test_is_fitted_property(self):
        """Test is_fitted property."""

        class TestEstimator(BaseEstimator):
            pass

        est = TestEstimator()
        assert est.is_fitted is False
        est.fit(None)
        assert est.is_fitted is True

    def test_check_is_fitted(self):
        """Test _check_is_fitted raises error when not fitted."""

        class TestEstimator(BaseEstimator):
            def predict(self):
                self._check_is_fitted()

        est = TestEstimator()
        with pytest.raises(ValueError, match="not fitted"):
            est.predict()


class TestBaseTransformer:
    """Tests for BaseTransformer."""

    def test_fit_transform(self):
        """Test fit_transform method."""

        class TestTransformer(BaseTransformer):
            def transform(self, y, X=None):
                import numpy as np
                return np.array(y) * 2

        trans = TestTransformer()
        result = trans.fit_transform([1, 2, 3])
        import numpy as np
        assert np.array_equal(result, np.array([2, 4, 6]))
        assert trans.is_fitted is True


class TestBaseForecaster:
    """Tests for BaseForecaster."""

    def test_predict_not_implemented(self):
        """Test predict raises NotImplementedError."""

        class TestForecaster(BaseForecaster):
            pass

        forecaster = TestForecaster()
        forecaster.fit([1, 2, 3])
        with pytest.raises(NotImplementedError):
            forecaster.predict(fh=1)

