import numpy as np
from core.physics import coherence, probability_zero
from core.sim import Noise, simulate, total_time_us
from core.crb import local_sigma
from est.bayes import posterior


def test_ramsey_limits():
    assert np.isclose(probability_zero(0, 0, 10, 1, .9), .95)
    assert np.isclose(coherence(10, 10), np.exp(-1))


def test_simulation_shapes_and_time():
    counts, n = simulate(.7, Noise(), 4, np.random.default_rng(2))
    assert counts.shape == n.shape == (6,)
    assert np.all(counts <= n)
    assert total_time_us(1) == 375


def test_repeat_counts_sum_to_pooled_counts():
    counts, _, repeats = simulate(.7, Noise(), 4, np.random.default_rng(4),
                                   return_repeat_counts=True)
    assert repeats.shape == (4, 6)
    assert np.array_equal(counts, repeats.sum(axis=0))


def test_local_precision_is_positive():
    assert local_sigma(.7, Noise(), 4) > 0


def test_shape_marginalized_posterior_is_normalized():
    counts, n = simulate(.7, Noise(shape=2.0), 1, np.random.default_rng(3))
    assert np.isclose(posterior(counts, n, "nuisance_shape").sum(), 1.0)
