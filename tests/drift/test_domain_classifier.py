import numpy as np
import pytest

from mercury.monitoring.drift.domain_classifier_drift_detector import DomainClassifierDrift


def test_domain_classifier():
    # Create Dataset with two variables. Create target dataset with drift and one with no drift
    X_src = np.array([np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)]).T
    X_trg_1 = np.array([np.random.normal(0, 1, 500), np.random.normal(0, 1, 500)]).T
    X_trg_2 = np.array([np.random.normal(0, 1, 500) + 2, np.random.normal(0, 1, 500)]).T

    # No drift case
    domain_class_drift_1 = DomainClassifierDrift(X_src, X_trg_1, features=["f1", "f2"], p_val=0.01, test_size=0.3,
                                                 n_runs=20, alpha=0.05)
    drift_metrics_1 = domain_class_drift_1.calculate_drift()
    assert drift_metrics_1["drift_detected"] == False
    assert len(drift_metrics_1["scores"]) == 2
    assert 'p_val' in drift_metrics_1.keys()

    # Drift case
    domain_class_drift_2 = DomainClassifierDrift(X_src, X_trg_2, features=["f1", "f2"], p_val=0.01, test_size=0.3,
                                                 n_runs=10)
    drift_metrics_2, df = domain_class_drift_2.calculate_drift(return_drift_score_target=True)
    assert drift_metrics_2["drift_detected"] == True
    assert df.shape[1] == 3

    # Plotting doesn't crash
    domain_class_drift_2.plot_feature_drift_scores()

    # Get drifted features not available in domain classifier
    with pytest.raises(Exception):
        domain_class_drift_2.get_drifted_features()

    assert drift_metrics_2["score"] > drift_metrics_1["score"]


def test_domain_classifier_exceptions():
    histograms_src = [np.array([5, 5, 5]), np.array([1, 2, 3])]
    histograms_target = [np.array([1, 1, 1]), np.array([1, 2, 3])]

    # Dataset not specified
    with pytest.raises(Exception):
        domain_class_drift = DomainClassifierDrift(features=["f1", "f2"], p_val=0.01, correction="bonferroni")
        drift_metrics = domain_class_drift.calculate_drift()

    # Histograms specify instead of dataset
    with pytest.raises(Exception):
        domain_class_drift = DomainClassifierDrift(distr_src=histograms_src, distr_target=histograms_target,
                                                   features=["f1", "f2"], p_val=0.01, correction="bonferroni")
        drift_metrics = domain_class_drift.calculate_drift()

    # Bad shape
    with pytest.raises(Exception):
        domain_class_drift = DomainClassifierDrift(X_src=np.zeros((3, 3, 3)), X_target=np.zeros((3, 3, 3)),
                                                   features=["f1", "f2"], p_val=0.01, correction="bonferroni")
        drift_metrics = domain_class_drift.calculate_drift()

    # X_src and X_target different number of features
    with pytest.raises(Exception):
        domain_class_drift = DomainClassifierDrift(X_src=np.zeros((5, 2)), X_target=np.zeros((5, 3)),
                                                   features=["f1", "f2"], p_val=0.01, correction="bonferroni")
        drift_metrics = domain_class_drift.calculate_drift()