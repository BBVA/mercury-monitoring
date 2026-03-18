import numpy as np
import pytest

from mercury.monitoring.drift.density_drift_detector import DensityDriftDetector


def test_density_detector():
    # Create datasets
    X_src = np.random.normal(0, 1, (20_000, 20))
    X_target_drifted = np.random.normal(2, 2, (10_000, 20))

    # We only test E2E functionality due to stochastic nature of the
    # method
    detector = DensityDriftDetector().fit(X_src)
    densities = detector.predict(X_target_drifted)
    assert densities.shape[0] == X_target_drifted.shape[0]

    # without early stopping callback
    detector = DensityDriftDetector().fit(X_src, epochs=3)


def test_explanations():
    # Create datasets
    X_src = np.random.normal(0, 1, (20_000, 20))
    X_target_drifted = X_src * 2
    detector = DensityDriftDetector().fit(X_src)
    deltas = detector.explain(X_target_drifted)

    assert deltas.shape == X_target_drifted.shape
    assert deltas.max() > 0  # some delta must be present

    # custom ref point
    deltas = detector.explain(X_target_drifted, ref_point=np.array([10, 10]))
    assert deltas.shape == X_target_drifted.shape


def test_density_detector_error_paths():
    with pytest.raises(ValueError):
        DensityDriftDetector(encoder=object())

    detector = DensityDriftDetector()
    detector.surrogate = None
    with pytest.raises(RuntimeError):
        detector._fit_surrogate()


if __name__ == "__main__":
    test_density_detector()
    test_explanations()
    test_density_detector_error_paths()
