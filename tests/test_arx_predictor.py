"""Tests for ARX predictor model."""
import numpy as np
import pytest

from dbs_bench.models.arx_predictor import ARXModel


def test_arx_fit_and_predict():
    """ARX model fits and predicts correctly."""
    n_state = 10
    horizon = 3
    n_samples = 200

    rng = np.random.default_rng(42)
    x = rng.standard_normal((n_samples, n_state))
    u = rng.standard_normal((n_samples, horizon)) * 0.01
    # Linear ground truth
    W_true = rng.standard_normal(n_state) * 0.1
    y = np.zeros((n_samples, horizon))
    for k in range(horizon):
        y[:, k] = x @ W_true + rng.standard_normal(n_samples) * 0.01

    model = ARXModel(n_state=n_state, horizon=horizon)
    metrics = model.fit_all(x, u, y)
    assert len(metrics) == horizon
    assert all(m["r2"] > 0.5 for m in metrics)

    # Single prediction
    pred = model.predict(x[0], u[0], k=1)
    assert np.isfinite(pred)

    # Batch prediction
    preds = model.predict_all(x[:5], u[:5])
    assert preds.shape == (5, horizon)


def test_arx_save_load(tmp_path):
    """ARX model round-trips through save/load."""
    model = ARXModel(n_state=5, horizon=2)
    rng = np.random.default_rng(0)
    x = rng.standard_normal((50, 5))
    u = rng.standard_normal((50, 2))
    y = rng.standard_normal((50, 2))
    model.fit_all(x, u, y)

    path = tmp_path / "arx.json"
    model.save(path)

    loaded = ARXModel.load(path)
    assert loaded.n_state == 5
    assert loaded.horizon == 2
    assert loaded.is_fitted

    # Predictions should match
    z = rng.standard_normal(5)
    u_test = rng.standard_normal(2)
    np.testing.assert_allclose(
        model.predict_all(z, u_test),
        loaded.predict_all(z, u_test),
    )
