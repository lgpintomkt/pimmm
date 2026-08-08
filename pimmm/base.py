import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted
from .integration import simulate_pimmm


class PhysicsInformedMMM(BaseEstimator, RegressorMixin):
    """
    Scikit-Learn Compatible Physics-Informed Marketing Mix Model.

    Combines machine learning response surface estimations of latent effort
    with the Generalized Innovation-Diffusion (GID) state equation.

    Parameters
    ----------
    response_model : object, default=None
        An optional base estimator (scikit-learn model, TabFM wrapper, or Meridian wrapper)
        estimating the response surface / latent marketing effort.
    dt : float, default=0.5
        Time increment step size for numerical ODE integration.
    gamma : float, default=1.0
        Fixed marketing scale factor. Treated as a hyperparameter (not optimized).
    p_bounds : tuple, default=(0.001, 0.1)
        Innovation coefficient (p) bounds.
    q_bounds : tuple, default=(0.1, 0.8)
        Imitation coefficient (q) bounds.
    F0 : float, default=0.001
        Initial adoption level used when fitting / predicting full trajectories.
    """

    def __init__(
        self,
        response_model=None,
        dt=0.5,
        gamma=1.0,
        p_bounds=(0.001, 0.1),
        q_bounds=(0.1, 0.8),
        F0=0.001,
    ):
        self.response_model = response_model
        self.dt = dt
        self.gamma = gamma
        self.p_bounds = p_bounds
        self.q_bounds = q_bounds
        self.F0 = F0

    def fit(self, X, y, t_eval=None):
        """
        Fit the PI-MMM parameters (p, q) against observed penetration y.
        gamma is treated as a fixed hyperparameter.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)

        if t_eval is None:
            t_eval = np.arange(0, len(y_arr) * self.dt, self.dt)

        # Extract spend signal
        if X_arr.ndim == 1 or X_arr.shape[1] == 1:
            spend_signal = X_arr.flatten()
        else:
            if self.response_model is not None:
                self.response_model.fit(X, y)
                spend_signal = self.response_model.predict(X)
            else:
                spend_signal = np.mean(X_arr, axis=1)

        def loss_function(params):
            p, q = params
            preds = simulate_pimmm(
                params=[p, q, self.gamma],
                spend_array=spend_signal,
                t_eval=t_eval,
                F0=self.F0,
                dt=self.dt,
            )
            return np.sum((preds - y_arr) ** 2)

        initial_guess = [0.01, 0.2]
        bounds = [self.p_bounds, self.q_bounds]

        res = minimize(
            loss_function, x0=initial_guess, bounds=bounds, method="L-BFGS-B"
        )

        self.p_opt_, self.q_opt_ = res.x
        self.is_fitted_ = True
        self.spend_signal_ = spend_signal

        # Store terminal state and training horizon for forecasting
        fitted_curve = simulate_pimmm(
            params=[self.p_opt_, self.q_opt_, self.gamma],
            spend_array=spend_signal,
            t_eval=t_eval,
            F0=self.F0,
            dt=self.dt,
        )
        self.F_end_ = float(fitted_curve[-1])
        self.t_end_ = float(t_eval[-1])

        return self

    def predict(self, X, t_eval=None):
        """
        Predict a full adoption trajectory starting from F0.
        Use this for in-sample curves or complete simulations.
        """
        check_is_fitted(self, attributes=["is_fitted_"])

        X_arr = np.asarray(X)
        if t_eval is None:
            t_eval = np.arange(0, len(X_arr) * self.dt, self.dt)

        if X_arr.ndim == 1 or X_arr.shape[1] == 1:
            spend_signal = X_arr.flatten()
        else:
            spend_signal = self.response_model.predict(X)

        params = [self.p_opt_, self.q_opt_, self.gamma]
        return simulate_pimmm(
            params, spend_signal, t_eval, F0=self.F0, dt=self.dt
        )

    def forecast(self, X_future, t_eval_future=None):
        """
        Continue the adoption curve from the end of the training period.

        This is the method that should be used for future-only forecasting
        so that the trajectory remains continuous.
        """
        check_is_fitted(self, attributes=["is_fitted_", "F_end_", "t_end_"])

        X_arr = np.asarray(X_future)

        if t_eval_future is None:
            n_steps = len(X_arr)
            t_eval_future = np.arange(
                self.t_end_ + self.dt,
                self.t_end_ + (n_steps + 1) * self.dt,
                self.dt,
            )[:n_steps]

        if X_arr.ndim == 1 or X_arr.shape[1] == 1:
            spend_signal = X_arr.flatten()
        else:
            spend_signal = self.response_model.predict(X_future)

        params = [self.p_opt_, self.q_opt_, self.gamma]
        return simulate_pimmm(
            params,
            spend_signal,
            t_eval_future,
            F0=self.F_end_,
            dt=self.dt,
        )

    def get_params_summary(self):
        """Returns recovered system physics constants (p, q) and the fixed gamma."""
        check_is_fitted(self, attributes=["is_fitted_"])
        return {
            "Innovation (p)": self.p_opt_,
            "Imitation (q)": self.q_opt_,
            "Scale Factor (gamma)": self.gamma,
        }