import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class TabFMResponseWrapper(BaseEstimator, RegressorMixin):
    """
    Adapter wrapper mapping Tabular Foundation Models (TabFM) 
    to act as an latent response surface driver for PhysicsInformedMMM.
    """
    def __init__(self, tabfm_model, mode="auto"):
        """
        mode: "auto" prefer regressor if available, "regressor" force regressor, "classifier" force classifier
        """
        self.tabfm_model = tabfm_model
        self.mode = mode
        self._validated = False

    def fit(self, X, y):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        # Prefer regression API if available
        if self.mode in ("auto", "regressor") and hasattr(self.tabfm_model, "fit") and hasattr(self.tabfm_model, "predict"):
            # Assume regressor API
            try:
                self.tabfm_model.fit(X_df, np.asarray(y))
            except Exception as e:
                raise RuntimeError(f"TabFM regressor fit failed: {e!r}")
            self._validated = True
            self._mode_used = "regressor"
            return self
        # Otherwise require classifier mode with explicit semantics
        if self.mode in ("auto", "classifier") and hasattr(self.tabfm_model, "fit") and hasattr(self.tabfm_model, "predict_proba"):
            # Validate classes and non-degenerate target
            if np.all(np.asarray(y) == y[0]):
                raise TabFMCompatibilityError("Target is constant; classifier mode is not appropriate.")
            # User must supply class mapping if they want classifier mode; do not auto-median-split
            raise TabFMCompatibilityError(
                "TabFM model exposes classifier API. To use classifier mode, provide pre-binned class labels "
                "and set mode='classifier'. Do not rely on automatic median-split conversion."
            )
        raise TabFMCompatibilityError("tabfm_model does not expose a compatible regression or classifier API.")

    def predict(self, X):
        if not self._validated:
            raise TabFMCompatibilityError("TabFMResponseWrapper not validated. Call fit(X, y) first.")
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        if self._mode_used == "regressor":
            preds = self.tabfm_model.predict(X_df)
            preds = np.asarray(preds).ravel()
            # Validate numeric outputs
            if not np.issubdtype(preds.dtype, np.number):
                raise TabFMCompatibilityError("Regressor returned non-numeric predictions.")
            return preds
        # classifier branch would require explicit mapping; omitted unless user requests classifier mode
        raise TabFMCompatibilityError("Classifier mode not enabled. Use regressor API or set mode='classifier' with explicit mapping.")

class MeridianCompatibilityError(TypeError):
    pass

class MeridianResponseWrapper(BaseEstimator, RegressorMixin):
    """
    Adapter wrapper mapping Google Meridian MMM outputs 
    into latent effort signals driver for PhysicsInformedMMM.
    """
    def __init__(self, meridian_model=None):
        self.meridian_model = meridian_model
        self._validated = False

    def fit(self, X, y=None):
        # Validate model API and fitted state if provided
        if self.meridian_model is None:
            raise MeridianCompatibilityError("MeridianResponseWrapper requires a meridian_model instance.")
        if not hasattr(self.meridian_model, "predict_adstock"):
            raise MeridianCompatibilityError(
                "Provided meridian_model does not implement predict_adstock(X). "
                "Ensure you are using a compatible Meridian version."
            )
        X_sample = np.asarray(X[:2]) if hasattr(X, "__len__") else np.asarray(X)
        try:
            out = self.meridian_model.predict_adstock(X_sample)
        except Exception as e:
            raise RuntimeError(f"meridian_model.predict_adstock raised during validation: {e!r}")
        out = np.asarray(out)
        if out.ndim < 1 or out.shape[0] != len(X_sample):
            raise MeridianCompatibilityError("predict_adstock returned unexpected shape.")
        self._validated = True
        return self

    def predict(self, X):
        if not self._validated:
            raise MeridianCompatibilityError("MeridianResponseWrapper not validated. Call fit(X, y) first.")
        try:
            adstock_response = self.meridian_model.predict_adstock(X)
        except AttributeError:
            # This is discovery-time error; re-raise as compatibility error
            raise MeridianCompatibilityError("meridian_model missing predict_adstock method.")
        except Exception as e:
            # Propagate runtime errors from the model
            raise RuntimeError(f"meridian_model.predict_adstock failed: {e!r}")
        arr = np.asarray(adstock_response)
        # Validate output shape and semantics
        if arr.ndim == 1:
            return arr
        return arr.mean(axis=-1)