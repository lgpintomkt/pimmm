import numpy as np
import pytest
from pimmm.base import PhysicsInformedMMM
from pimmm.integration import simulate_pimmm


def test_gamma_is_fixed_hyperparameter():
    """gamma is not optimized; only p and q are free parameters."""
    t = np.arange(0.0, 20.0, 0.5)
    spend = np.linspace(1.0, 4.0, len(t))

    # Ground-truth trajectory with known p, q and gamma=1.0
    true_p, true_q, true_gamma = 0.03, 0.45, 1.0
    y = simulate_pimmm([true_p, true_q, true_gamma], spend, t)

    model = PhysicsInformedMMM(gamma=1.0, dt=0.5)
    model.fit(spend, y, t_eval=t)

    # gamma must stay exactly at the value we set
    assert model.gamma == 1.0
    assert model.get_params_summary()["Scale Factor (gamma)"] == 1.0

    # Fitted curve should match the observed data well
    preds = model.predict(spend, t_eval=t)
    mse = np.mean((preds - y) ** 2)
    assert mse < 0.01

    # Recovered parameters should be in a reasonable range
    assert 0.001 <= model.p_opt_ <= 0.1
    assert 0.1 <= model.q_opt_ <= 0.8


def test_different_gamma_produces_different_curves():
    """Changing the fixed gamma must change the simulated trajectory."""
    t = np.arange(0.0, 10.0, 0.5)
    spend = np.linspace(1.0, 3.0, len(t))
    params = [0.02, 0.4]

    curve_g1 = simulate_pimmm(params + [1.0], spend, t)
    curve_g05 = simulate_pimmm(params + [0.5], spend, t)

    assert np.max(np.abs(curve_g1 - curve_g05)) > 0.01


def test_old_non_identifiability_is_gone():
    """
    With gamma fixed, the two previously equivalent triples
    (0.02, 0.4, 0.8) and (0.04, 0.8, 0.4) are no longer equivalent.
    """
    t = np.arange(0.0, 10.0, 0.5)
    spend = np.linspace(1.0, 3.0, len(t))

    # Force the same gamma → the trajectories must now differ
    curve_a = simulate_pimmm([0.02, 0.4, 0.8], spend, t)
    curve_b = simulate_pimmm([0.04, 0.8, 0.8], spend, t)  # same gamma

    assert np.max(np.abs(curve_a - curve_b)) > 0.05


def test_predict_uses_fixed_gamma():
    """predict() must respect the gamma that was set at construction."""
    t = np.arange(0.0, 15.0, 0.5)
    spend = np.linspace(0.5, 2.5, len(t))
    y = simulate_pimmm([0.025, 0.35, 0.7], spend, t)

    model = PhysicsInformedMMM(gamma=0.7, dt=0.5)
    model.fit(spend, y, t_eval=t)

    preds = model.predict(spend, t_eval=t)
    mse = np.mean((preds - y) ** 2)
    assert mse < 0.01
