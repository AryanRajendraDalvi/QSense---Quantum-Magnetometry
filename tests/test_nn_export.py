"""Test that numpy forward pass matches torch within tolerance."""
from __future__ import annotations
import pytest
import numpy as np
from pathlib import Path

WEIGHTS = Path("artifacts/nn_weights.npz")


@pytest.mark.skipif(not WEIGHTS.exists(), reason="nn_weights.npz not yet trained")
def test_numpy_torch_equivalence():
    """Numpy forward pass must match torch within 1e-5."""
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        pytest.skip("torch not installed")

    from train.train_nn import QSenseNet
    from est.nn_infer import QSenseNetNumpy

    rng = np.random.default_rng(77)
    x_np = rng.uniform(0, 1, (500, 19)).astype(np.float32)
    x_th = torch.from_numpy(x_np)

    state = {k: torch.from_numpy(v) for k, v in np.load(str(WEIGHTS)).items()}
    model = QSenseNet()
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        torch_probs = F.softmax(model(x_th), dim=-1).numpy()

    np_model = QSenseNetNumpy(WEIGHTS)
    np_probs = np_model.forward(x_np.astype(np.float64))

    max_diff = np.abs(torch_probs - np_probs).max()
    assert max_diff < 1e-5, f"Max |torch - numpy| = {max_diff:.2e} > 1e-5"


@pytest.mark.skipif(not WEIGHTS.exists(), reason="nn_weights.npz not yet trained")
def test_posterior_sums_to_one():
    """NN posterior must sum to 1 within floating point."""
    from est.nn_infer import posterior
    counts = np.array([4, 5, 6, 7, 5, 4])
    n = np.array([10, 10, 10, 10, 10, 10])
    mass = posterior(counts, n)
    assert abs(mass.sum() - 1.0) < 1e-6, f"Posterior sum = {mass.sum()}"
    assert np.all(mass >= 0), "Posterior has negative values"


@pytest.mark.skipif(not WEIGHTS.exists(), reason="nn_weights.npz not yet trained")
def test_feature_extraction_shapes():
    """Feature extraction returns (19,) for standard inputs."""
    from est.nn_infer import extract_features
    counts = np.array([4, 5, 6, 7, 5, 4])
    n = np.array([40, 40, 40, 40, 40, 40])  # k=4, 10 shots each
    rc = np.array([[1,2,1,2,1,1],[2,1,2,2,1,1],[1,2,2,1,2,2],[0,0,1,2,1,0]])
    feat = extract_features(counts, n, repeat_counts=rc)
    assert feat.shape == (19,), f"Expected (19,) but got {feat.shape}"
    assert feat.dtype == np.float32
