import numpy as np
import pytest
from types import SimpleNamespace
from pimmm.base import PhysicsInformedMMM


# --- Helpers ---------------------------------------------------------------

def make_res(x, fun, success=True, status=0, message="ok"):
    return SimpleNamespace(x=np.asarray(x), fun=float(fun), success=success, status=status, message=message)


# --- Tests -----------------------------------------------------------------

def test_rejects_nan_in_y():
    model = PhysicsInformedMMM()
    X = np.linspace(1.0, 2.0, 5)
    y = np.array([0.1, np.nan, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        model.fit(X, y)


def test_t_eval_length_and_order_validation():
    model = PhysicsInformedMMM()
    X = np.linspace(1.0, 2.0, 5)
    y = np.linspace(0.1, 0.5, 5)

    # mismatched length
    t_eval_bad_len = np.arange(0, 3 * model.dt, model.dt)
    with pytest.raises(ValueError):
        model.fit(X, y, t_eval=t_eval_bad_len)

    # non-increasing
    t_eval_non_increasing = np.array([0.0, 0.5, 0.5, 1.5, 2.0])
    with pytest.raises(ValueError):
        model.fit(X, y, t_eval=t_eval_non_increasing)


def test_optimization_failure_raises_and_not_fitted(monkeypatch):
    model = PhysicsInformedMMM()
    X = np.linspace(1.0, 2.0, 5)
    y = np.linspace(0.1, 0.5, 5)

    # minimize returns a result that indicates failure
    def fake_minimize(*args, **kwargs):
        return make_res([0.01, 0.2], fun=np.inf, success=False, status=1, message="failed")

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    with pytest.raises(RuntimeError):
        model.fit(X, y)

    assert not getattr(model, "is_fitted_", False)


def test_nonfinite_result_raises(monkeypatch):
    model = PhysicsInformedMMM()
    X = np.linspace(1.0, 2.0, 5)
    y = np.linspace(0.1, 0.5, 5)

    # minimize returns success True but non-finite parameters
    def fake_minimize(*args, **kwargs):
        return make_res([np.inf, 0.2], fun=1.0, success=True)

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    with pytest.raises(RuntimeError):
        model.fit(X, y)

    assert not getattr(model, "is_fitted_", False)


def test_successful_fit_publishes_attributes(monkeypatch):
    model = PhysicsInformedMMM()
    X = np.linspace(1.0, 2.0, 5)
    y = np.linspace(0.01, 0.05, 5)

    # minimize returns a valid successful result
    def fake_minimize(*args, **kwargs):
        return make_res([0.02, 0.25], fun=0.001, success=True)

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    fitted = model.fit(X, y)
    assert fitted is model
    assert getattr(model, "is_fitted_", False) is True
    assert np.isfinite(model.p_opt_)
    assert np.isfinite(model.q_opt_)
    assert hasattr(model, "optimization_result_")
    assert np.isfinite(model.optimization_result_.fun)
    assert np.isfinite(model.F_end_)
    assert np.isfinite(model.t_end_)


def test_multiple_restarts_selects_best(monkeypatch):
    model = PhysicsInformedMMM()
    X = np.linspace(1.0, 2.0, 6)
    y = np.linspace(0.01, 0.06, 6)

    # Create a sequence of results with different fun values
    results = [
        make_res([0.05, 0.3], fun=0.5, success=True),
        make_res([0.02, 0.25], fun=0.01, success=True),
        make_res([0.03, 0.2], fun=0.1, success=True),
    ]
    calls = {"i": 0}

    def fake_minimize(*args, **kwargs):
        i = calls["i"]
        calls["i"] += 1
        # cycle through prepared results; if more calls than results, return worst
        return results[i] if i < len(results) else results[0]

    monkeypatch.setattr("scipy.optimize.minimize", fake_minimize)

    model.fit(X, y, n_restarts=3)
    assert model.optimization_result_.fun == pytest.approx(0.01)
    assert np.isfinite(model.p_opt_)
    assert np.isfinite(model.q_opt_)
    assert model.is_fitted_ is True
