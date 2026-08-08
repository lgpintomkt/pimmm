import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from .integration import simulate_pimmm

class PhysicsInformedMMM(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        response_model=None,
        dt=0.5,
        gamma=1.0,                    # now a fixed hyperparameter
        p_bounds=(0.001, 0.1),
        q_bounds=(0.1, 0.8),
        F0=0.001
    ):
        self.response_model = response_model
        self.dt = dt
        self.gamma = gamma
        self.p_bounds = p_bounds
        self.q_bounds = q_bounds
        self.F0 = F0

    def fit(self, X, y, t_eval=None):
        def loss_function(params):
            p, q = params
            preds = simulate_pimmm(
                params=[p, q, self.gamma],
                spend_array=spend_signal,
                t_eval=t_eval,
                F0=self.F0,
                dt=self.dt
            )
            return np.sum((preds - y_arr) ** 2)

        initial_guess = [0.01, 0.2]
        bounds = [self.p_bounds, self.q_bounds]

        res = minimize(loss_function, x0=initial_guess, bounds=bounds, method='L-BFGS-B')

        self.p_opt_, self.q_opt_ = res.x
        self.is_fitted_ = True
        self.spend_signal_ = spend_signal
        return self

    def predict(self, X_future, t_eval_future=None):
        params = [self.p_opt_, self.q_opt_, self.gamma]
        return simulate_pimmm(params, spend_signal, t_eval_future, F0=self.F0, dt=self.dt)

    def get_params_summary(self):
        check_is_fitted(self, attributes=["is_fitted_"])
        return {
            "Innovation (p)": self.p_opt_,
            "Imitation (q)": self.q_opt_
        }