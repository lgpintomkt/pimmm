import numpy as np
import pytest
from pimmm.base import PhysicsInformedMMM
from pimmm.integration import simulate_pimmm


def test_forecast_is_continuous():
    """Forecast must start exactly from the last fitted state (no jump)."""
    dt = 0.5
    t = np.arange(0.0, 20.0, dt)
    spend = 2.0 + np.sin(0.2 * t)
    y = simulate_pimmm([0.02, 0.4, 1.0], spend, t, F0=0.01, dt=dt)

    cutoff = 20  # first 20 points for training
    model = PhysicsInformedMMM(dt=dt, F0=0.01, gamma=1.0)
    model.fit(spend[:cutoff], y[:cutoff], t_eval=t[:cutoff])

    last_training = model.predict(spend[:cutoff], t_eval=t[:cutoff])[-1]
    first_future = model.forecast(spend[cutoff:])[0]

    assert first_future == pytest.approx(last_training, abs=1e-5)
    assert first_future == pytest.approx(model.F_end_, abs=1e-5)


def test_forecast_uses_future_spend():
    """Different future spend should produce different forecasts."""
    dt = 0.5
    t = np.arange(0.0, 15.0, dt)
    spend = np.linspace(1.0, 3.0, len(t))
    y = simulate_pimmm([0.03, 0.4, 1.0], spend, t, F0=0.01, dt=dt)

    model = PhysicsInformedMMM(dt=dt, F0=0.01, gamma=1.0)
    model.fit(spend[:20], y[:20], t_eval=t[:20])

    high_spend = np.full(10, 5.0)
    low_spend = np.full(10, 0.5)

    forecast_high = model.forecast(high_spend)
    forecast_low = model.forecast(low_spend)

    assert forecast_high[-1] > forecast_low[-1]