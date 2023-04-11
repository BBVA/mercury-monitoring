from mercury.monitoring.drift.metrics import hellinger_distance
import numpy as np
import pytest

from pytest import approx


def test_hellinger():
    # Basic Tests
    p = np.array([1,1,0])
    q = np.array([0,1,1])
    dist1 = hellinger_distance(p,q)
    assert dist1 == 1.0

    # Distance in this test should be smaller than before
    p = np.array([1,1,0])
    q = np.array([2,1,0])
    dist2 = hellinger_distance(p,q)
    assert dist2 < dist1

def test_hellinger_max_distance():
    # Test gives max hellinger distance which is sqrt(2)
    p = np.array([1,0,1])
    q = np.array([0,1,0])

    assert approx(hellinger_distance(p,q), rel=0.001) == np.sqrt(2)

def test_hellinger_min_distance():
    # Test gives min hellinger distance which is 0
    p = np.array([1,1,1])
    q = np.array([10,10,10])

    assert hellinger_distance(p,q) == 0

def test_hellinger_normalize_false():
    p = np.array([0.5,0.5])
    q = np.array([0.5,0.5])

    assert hellinger_distance(p,q, normalize=False) == 0

def test_hellinger_exceptions():

    # Histograms with different lengths
    p = np.array([0.5, 0.5, 0.2])
    q = np.array([0.5, 0.5])
    with pytest.raises(Exception):
        hellinger_distance(p, q)
