import numpy as np
import pandas as pd
import pytest
from types import SimpleNamespace
from unittest.mock import Mock
from pimmm.wrappers import (
    MeridianResponseWrapper, MeridianCompatibilityError,
    TabFMResponseWrapper, TabFMCompatibilityError
)

# --- Helpers / fake models ---------------------------------------------------

class FakeMeridianGood:
    """Simulates a well-behaved meridian model returning adstocked channels."""
    def predict_adstock(self, X):
        X = np.asarray(X)
        # return shape (n_samples, n_channels) or (n_samples,)
        return np.stack([X.sum(axis=1), X.mean(axis=1)], axis=-1)

class FakeMeridianMissing:
    """No predict_adstock attribute (simulates older/incompatible API)."""
    pass

class FakeMeridianRaisesAttrInside:
    """Has predict_adstock but raises AttributeError internally (should propagate)."""
    def predict_adstock(self, X):
        # simulate an internal AttributeError (e.g., bug in implementation)
        raise AttributeError("internal bug")

class FakeTabFMRegressor:
    """Simulates a TabFM regressor API."""
    def __init__(self):
        self._fitted = False
    def fit(self, X, y):
        self._fitted = True
    def predict(self, X):
        if not self._fitted:
            raise RuntimeError("not fitted")
        X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return np.arange(len(X)) * 0.5  # deterministic numeric output

class FakeTabFMClassifier:
    """Simulates a TabFM classifier API with predictable classes_ and predict_proba."""
    def __init__(self, classes=("low", "high")):
        self.classes_ = np.array(classes)
        self._fitted = False
    def fit(self, X, y):
        self._fitted = True
        # store classes seen
        self.classes_ = np.unique(y).astype(str)
    def predict_proba(self, X):
        if not self._fitted:
            raise RuntimeError("not fitted")
        n = len(X)
        # return two-column probs where second column is "high"
        probs = np.vstack([np.linspace(1.0, 0.0, n), np.linspace(0.0, 1.0, n)]).T
        return probs

# --- Meridian tests ---------------------------------------------------------

def test_meridian_fit_requires_model_and_predict_adstock():
    wrapper = MeridianResponseWrapper(meridian_model=None)
    with pytest.raises(TypeError):
        wrapper.fit(np.zeros((2, 3)), None)

    wrapper2 = MeridianResponseWrapper(meridian_model=FakeMeridianMissing())
    with pytest.raises(TypeError):
        wrapper2.fit(np.zeros((2, 3)), None)

def test_meridian_validates_predict_adstock_return_shape():
    model = FakeMeridianGood()
    wrapper = MeridianResponseWrapper(meridian_model=model)
    # should not raise
    wrapper.fit(np.ones((3, 2)), None)
    preds = wrapper.predict(np.ones((3, 2)))
    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == 3

def test_meridian_does_not_swallow_internal_attribute_error():
    model = FakeMeridianRaisesAttrInside()
    wrapper = MeridianResponseWrapper(meridian_model=model)
    # validation should surface the internal AttributeError as a runtime error
    with pytest.raises(RuntimeError):
        wrapper.fit(np.ones((2, 2)), None)

def test_meridian_predict_requires_fit_first():
    model = FakeMeridianGood()
    wrapper = MeridianResponseWrapper(meridian_model=model)
    # calling predict before fit should raise a compatibility error
    with pytest.raises(TypeError):
        wrapper.predict(np.ones((2, 2)))

# --- TabFM tests ------------------------------------------------------------

def test_tabfm_prefers_regressor_api_and_predicts_numeric():
    model = FakeTabFMRegressor()
    wrapper = TabFMResponseWrapper(tabfm_model=model, mode="auto")
    X = np.random.randn(4, 3)
    y = np.array([0.1, 0.2, 0.3, 0.4])
    wrapper.fit(X, y)  # should choose regressor path
    preds = wrapper.predict(X)
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (4,)
    assert np.all(np.isfinite(preds))

def test_tabfm_rejects_constant_target_for_classifier_mode():
    model = FakeTabFMClassifier()
    wrapper = TabFMResponseWrapper(tabfm_model=model, mode="classifier")
    X = np.random.randn(5, 2)
    y_constant = np.ones(5)  # degenerate target
    with pytest.raises(TypeError):
        wrapper.fit(X, y_constant)

def test_tabfm_classifier_not_auto_median_split():
    # If user explicitly requests classifier mode, they must provide class labels.
    model = FakeTabFMClassifier()
    wrapper = TabFMResponseWrapper(tabfm_model=model, mode="classifier")
    X = np.random.randn(6, 2)
    # Provide explicit class labels (strings) to satisfy classifier contract
    y_classes = np.array(["low", "high", "low", "high", "low", "high"])
    wrapper.fit(X, y_classes)
    # If wrapper supports classifier mode mapping, predict should raise unless mapping exists.
    with pytest.raises(TypeError):
        wrapper.predict(X)

def test_tabfm_predict_requires_fit_first():
    model = FakeTabFMRegressor()
    wrapper = TabFMResponseWrapper(tabfm_model=model, mode="regressor")
    X = np.random.randn(3, 2)
    with pytest.raises(TypeError):
        wrapper.predict(X)

# --- Interaction tests / edge cases -----------------------------------------

def test_tabfm_regressor_fit_failure_propagates():
    bad_model = Mock()
    bad_model.fit.side_effect = RuntimeError("fit failed")
    wrapper = TabFMResponseWrapper(tabfm_model=bad_model, mode="regressor")
    with pytest.raises(RuntimeError):
        wrapper.fit(np.zeros((2, 2)), np.array([0.1, 0.2]))

def test_meridian_predict_runtime_error_propagates():
    # model that raises a runtime error inside predict_adstock
    class BadMeridian:
        def predict_adstock(self, X):
            raise RuntimeError("boom")
    wrapper = MeridianResponseWrapper(meridian_model=BadMeridian())
    with pytest.raises(RuntimeError):
        wrapper.fit(np.zeros((2, 2)), None)