from mercury.monitoring.drift.metrics import jeffreys_divergence
import numpy as np

import pytest

def test_jeffrey():
    # Basic Tests
    p = np.array([1,1,1])
    q = np.array([1,1,2])
    jeff_1 = jeffreys_divergence(p,q)

    p = np.array([1,1,1])
    q = np.array([1,1,10])
    jeff_2 = jeffreys_divergence(p,q)

    assert jeff_1 < jeff_2

def test_jeffrey_symmetry():
    p = np.array([1,2,1])
    q = np.array([2,1,2])
    jeff_1 = jeffreys_divergence(p,q)

    p = np.array([2,1,2])
    q = np.array([1,2,1])
    jeff_2 = jeffreys_divergence(p,q)

    assert jeff_1 == jeff_2


def test_jeffrey_min_distance():
    p = np.array([1,1,1])
    q = np.array([10,10,10])
    assert jeffreys_divergence(p,q) == 0

def test_jeffrey_normalize_false():
    p = np.array([0.33,0.33,0.33])
    q = np.array([0.33,0.33,0.33])
    assert jeffreys_divergence(p,q, normalize=False) == 0

def test_jeffrey_special_case_zeros():
    p = np.array([0,1,0])
    q = np.array([1,0,1])
    assert jeffreys_divergence(p,q) > 0

def test_jeffreys_exceptions():

    # Histograms with different lengths
    p = np.array([0.5, 0.5, 0.2])
    q = np.array([0.5, 0.5])
    with pytest.raises(Exception):
        jeffreys_divergence(p, q)
