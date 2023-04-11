import pytest
import numpy as np

from mercury.monitoring.drift.auto_encoder_drift_detector import AutoEncoderDriftDetector


def test_ae_detector():
    # Test reproducibility
    np.random.seed(10)

    # Create datasets
    X_src = np.random.normal(0, 1, (20_000, 20))

    X_target_drifted = np.random.normal(0, 1, (10_000, 20))
    X_target_drifted[:, 10] += 20

    X_target_nodrifted = np.random.normal(0, 1, (10_000, 20))

    # No drift case 1
    detector = AutoEncoderDriftDetector(p_val=.01).fit(X_src)
    result = detector.calculate_drift(X_target_nodrifted)
    assert result['drift'] == False

    # No drift case 2
    detector = AutoEncoderDriftDetector().fit(X_src)
    result = detector.calculate_drift(X_src)
    assert result['drift'] == False

    # Drift case
    detector = AutoEncoderDriftDetector().fit(X_src)
    result = detector.calculate_drift(X_target_drifted)
    assert result['drift'] == True

    # Drift case (custom metric)
    detector = AutoEncoderDriftDetector(custom_reconstruction_error='mae').fit(X_src)
    result = detector.calculate_drift(X_target_drifted)
    assert result['drift'] == True

    # Test contains extra info
    result = detector.calculate_drift(X_target_drifted, return_errors=True)

    assert 'source_errors' in result
    assert 'target_errors' in result

def test_ae_detector_custom_reconstruction_error():
    # Test the detector passing a lambda as reconstruction error

    # Create datasets
    X_src = np.random.normal(0, 1, (20_000, 20))
    X_target_drifted = np.random.normal(0, 1, (10_000, 20))
    X_target_drifted[:, 10] += 20

    mse = lambda x, y: (x-y)**2

    detector = AutoEncoderDriftDetector(custom_reconstruction_error=mse).fit(X_src)
    result = detector.calculate_drift(X_target_drifted)
    assert result['drift'] == True




def test_ae_detect_exceptions():
    with pytest.raises(ValueError):
        AutoEncoderDriftDetector(fitted_models=[1,2,3])

    with pytest.raises(ValueError):
        AutoEncoderDriftDetector(fitted_models=[1,2,3], reference_errors=[1,2])

    with pytest.raises(ValueError):
        AutoEncoderDriftDetector(autoencoder_kwargs='test')