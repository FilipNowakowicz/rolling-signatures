import numpy as np
import pytest

from sigtrade._backend import n_features, signature


@pytest.fixture
def path():
    rng = np.random.default_rng(0)
    return rng.standard_normal((10, 2)).cumsum(axis=0)


def test_level_one_is_net_displacement(path):
    sig = signature(path, depth=1)
    assert sig == pytest.approx(path[-1] - path[0])


def test_diagonal_level_two_is_half_squared_displacement(path):
    sig = signature(path, depth=2)
    s1 = sig[:2]
    s11, s22 = sig[2], sig[5]
    assert s11 == pytest.approx(0.5 * s1[0] ** 2)
    assert s22 == pytest.approx(0.5 * s1[1] ** 2)


def test_shuffle_identity(path):
    sig = signature(path, depth=2)
    s1, s2 = sig[0], sig[1]
    s11, s12, s21, s22 = sig[2], sig[3], sig[4], sig[5]
    assert s1 * s2 == pytest.approx(s12 + s21)
    assert s1 * s1 == pytest.approx(2 * s11)
    assert s2 * s2 == pytest.approx(2 * s22)


def test_short_path_is_zero():
    sig = signature(np.zeros((1, 2)), depth=3)
    assert sig == pytest.approx(np.zeros(n_features(2, 3)))


def test_backends_agree(path):
    iisignature = pytest.importorskip("iisignature")
    sig_rp = signature(path, depth=3, backend="roughpy")
    sig_ii = signature(path, depth=3, backend="iisignature")
    assert sig_rp == pytest.approx(sig_ii, abs=1e-8)
