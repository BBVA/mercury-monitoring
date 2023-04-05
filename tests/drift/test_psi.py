from mercury.monitoring.drift.metrics import psi
import numpy as np
import pytest

def test_psi():
    # Basic Test
    a = np.array([12, 11, 14, 12, 12, 10, 12, 6, 6, 5])
    b = np.array([11, 11, 12, 13, 11, 11, 13, 5, 7, 6])
    small_psi1 = psi(a, b, normalize=True)
    assert small_psi1 == pytest.approx(0.0125, 0.01)

    a = np.array([12, 4, 14, 2, 12, 1, 12, 1, 6, 1])
    b = np.array([6, 11, 5, 13, 6, 11, 6, 5, 1, 6])
    big_psi1 = psi(a, b, normalize=True)
    assert big_psi1 == pytest.approx(1.436, 0.01)

    assert big_psi1 > small_psi1

    # Autoinfer that should normalize
    big_psi2 = psi(a, b, normalize=None)
    assert big_psi1 == big_psi2

def test_psi_normalize_false():
    a = np.array([0.12, 0.11, 0.14, 0.12, 0.12, 0.1 , 0.12, 0.06, 0.06, 0.05])
    b = np.array([0.11, 0.11, 0.12, 0.13, 0.11, 0.11, 0.13, 0.05, 0.07, 0.06])
    psi1 = psi(a, b, normalize=False)
    assert psi1 == pytest.approx(0.0125, 0.01)

    # Autoinfer that it shouldn't normalize
    psi2 = psi(a, b, normalize=None)
    assert psi2 == psi1


def test_psi_special_case():
    a = np.array([0, 11, 0, 13, 12, 10, 12, 6, 6, 5])
    b = np.array([11, 0, 0, 13, 11, 11, 13, 5, 7, 6])
    psi1 = psi(a, b, normalize=True, eps=1e-4)
    assert psi1 == pytest.approx(2.115, 0.01)

    # With eps=0 we get nan value
    psi2 = psi(a, b, normalize=True, eps=0)
    assert np.isnan(psi2)

def test_psi_bad_inputs():
    a = np.array([[0, 11, 0, 13, 12, 10, 12, 6, 6, 5]])
    b = np.array([[11, 0, 0, 13, 11, 11, 13, 5, 7, 6]])

    with pytest.raises(ValueError):
        psi1 = psi(a, b, normalize=True, eps=1e-4)

    a = np.array([0, 11, 0, 13, 12])
    b = np.array([11, 0, 0, 13, 11, 11, 13, 5, 7, 6])

    with pytest.raises(ValueError):
        psi1 = psi(a, b, normalize=True, eps=1e-4)