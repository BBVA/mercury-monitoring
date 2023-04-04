
from mercury.monitoring.drift.metrics import bootstrap_eval_drift
from mercury.monitoring.drift.domain_classifier_drift_detector import DomainClassifierDrift
import numpy as np
import pytest

def test_bootstrap_eval_drift():

    def eval_acc(y_true, y_pred):
        return np.sum(y_true == y_pred) / y_true.shape[0]

    # no drift case
    y_true_src = np.array([0,0,0,0,0,1,1,1,1,1])
    y_pred_src = np.array([0,0,0,1,1,0,0,1,1,1])

    y_true_target = np.flip(y_true_src)
    y_pred_target = np.flip(y_pred_src)

    drift_metrics, drift_detector, dist_src, dist_target = bootstrap_eval_drift(
            y_true_src=y_true_src, y_pred_src=y_pred_src, y_true_target=y_true_target, y_pred_target=y_pred_target,
            resample_size_src=8, resample_size_target=8, num_resamples=1000,
            eval_fn=eval_acc, drift_detector=None
        )
    assert drift_metrics["drift_detected"] == False
    assert np.mean(dist_src) == pytest.approx(0.6, 0.1)
    assert np.mean(dist_target) == pytest.approx(0.6, 0.1)

    # drift case
    y_true_src = np.array([0,0,0,0,0,1,1,1,1,1])
    y_pred_src = np.array([0,0,0,0,1,0,1,1,1,1])

    y_true_target = np.array([0,0,0,0,0,1,1,1,1,1])
    y_pred_target = np.array([0,1,1,1,1,0,0,0,0,1])

    drift_metrics, drift_detector, dist_src, dist_target = bootstrap_eval_drift(
            y_true_src=y_true_src, y_pred_src=y_pred_src, y_true_target=y_true_target, y_pred_target=y_pred_target,
            resample_size_src=None, resample_size_target=None, num_resamples=1000,
            eval_fn=eval_acc, drift_detector=None
        )

    assert drift_metrics["drift_detected"] == True
    assert np.mean(dist_src) == pytest.approx(0.8, 0.1)
    assert np.mean(dist_target) == pytest.approx(0.2, 0.1)

    # Drift Case passing DomainClassifierDrift as drift dtector
    y_true_src = np.array([0,0,0,0,0,1,1,1,1,1])
    y_pred_src = np.array([0,0,0,0,1,0,1,1,1,1])

    y_true_target = np.array([0,0,0,0,0,1,1,1,1,1])
    y_pred_target = np.array([0,1,1,1,1,0,0,0,0,1])

    drift_detector = DomainClassifierDrift(features=["accuracy"], p_val=0.01, test_size=0.3, n_runs=10)

    drift_metrics, drift_detector, dist_src, dist_target = bootstrap_eval_drift(
            y_true_src=y_true_src, y_pred_src=y_pred_src, y_true_target=y_true_target, y_pred_target=y_pred_target,
            resample_size_src=8, resample_size_target=8, num_resamples=1000,
            eval_fn=eval_acc, drift_detector=drift_detector
        )
    assert drift_metrics["drift_detected"] == True
    assert np.mean(dist_src) == pytest.approx(0.8, 0.1)
    assert np.mean(dist_target) == pytest.approx(0.2, 0.1)