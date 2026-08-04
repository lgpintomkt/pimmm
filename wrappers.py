import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class TabFMResponseWrapper(BaseEstimator, RegressorMixin):
    """
    Adapter wrapper mapping Tabular Foundation Models (TabFM) 
    to act as an latent response surface driver for PhysicsInformedMMM.
    """
    def __init__(self, tabfm_model):
        self.tabfm_model = tabfm_model
        
    def fit(self, X, y):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        # Adapt continuous target into binary quantile bins for TabFM underlying classifier
        y_class = (y > np.median(y)).astype(str)
        self.tabfm_model.fit(X_df, y_class)
        return self

    def predict(self, X):
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        probs = self.tabfm_model.predict_proba(X_df)
        # Extract response intensity from probability distribution
        if probs.ndim > 1:
            return probs[:, -1]
        return probs


class MeridianResponseWrapper(BaseEstimator, RegressorMixin):
    """
    Adapter wrapper mapping Google Meridian MMM outputs 
    into latent effort signals driver for PhysicsInformedMMM.
    """
    def __init__(self, meridian_model=None):
        self.meridian_model = meridian_model

    def fit(self, X, y):
        # Meridian fit steps managed externally via Meridian object pipeline
        return self

    def predict(self, X):
        if self.meridian_model is not None:
            # Extract latent media response adstock transformed outputs from Meridian
            try:
                adstock_response = self.meridian_model.predict_adstock(X)
                return np.asarray(adstock_response).mean(axis=-1)
            except AttributeError:
                pass
        
        # Fallback to mean channel intensity
        X_arr = np.asarray(X)
        return np.mean(X_arr, axis=1) if X_arr.ndim > 1 else X_arr