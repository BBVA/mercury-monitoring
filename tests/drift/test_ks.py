import numpy as np
import pytest

from mercury.monitoring.drift.ks_drift_detector import KSDrift


def test_ks():
    # Create Dataset with two variables. Create target dataset with drift and one with no drift
    X_src = np.array([np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)]).T
    X_trg_1 = np.array([np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)]).T
    X_trg_2 = np.array([np.random.normal(0, 1, 1000) + 2, np.random.normal(0, 1, 1000)]).T

    # No drift case
    ks_drift_1 = KSDrift(X_src=X_src, X_target=X_trg_1, features=["f1", "f2"], p_val=0.0001, correction="bonferroni")
    drift_metrics_1 = ks_drift_1.calculate_drift()
    assert drift_metrics_1["drift_detected"] == False
    assert len(drift_metrics_1["p_vals"]) == 2
    assert len(drift_metrics_1["scores"]) == 2
    assert drift_metrics_1["threshold"] == (0.0001 / X_src.shape[1])

    # Drift case
    ks_drift_2 = KSDrift(X_src=X_src, X_target=X_trg_2, features=["f1", "f2"], p_val=0.05, correction="bonferroni")
    drift_metrics_2 = ks_drift_2.calculate_drift()
    assert drift_metrics_2["drift_detected"] == True
    assert drift_metrics_2["scores"][0] > drift_metrics_2["scores"][1]
    assert 'f1' in ks_drift_2.get_drifted_features()

    # score of drift case is higher than no drift case
    assert drift_metrics_2["score"] > drift_metrics_1["score"]

    # Plotting doesn't crash
    ks_drift_2.plot_distribution_drifted_features()
    ks_drift_2.plot_feature_drift_scores()

    # Set back the first target dataset
    ks_drift_2.set_datasets(X_src, X_trg_1)
    ks_drift_2.p_val = 0.0001
    assert ks_drift_2.drift_metrics is None
    drift_metrics = ks_drift_2.calculate_drift()
    assert drift_metrics["drift_detected"] == 0

    # Try without correction
    ks_drift_1 = KSDrift(X_src=X_src, X_target=X_trg_1, features=["f1", "f2"], p_val=0.01, correction=None)
    drift_metrics_1 = ks_drift_1.calculate_drift()
    assert drift_metrics_1["threshold"] == 0.01

    # Test when features are not specified:
    ks_drift_3 = KSDrift(X_src=X_src, X_target=X_trg_2, p_val=0.01, correction="bonferroni")
    drift_metrics_3 = ks_drift_3.calculate_drift()
    # Plotting doesn't crash
    ks_drift_3.plot_distribution_drifted_features()
    ks_drift_3.plot_feature_drift_scores(top_k=1)
    assert ks_drift_3.get_drifted_features(return_as_indices=True) == [0]
    with pytest.raises(Exception):
        ks_drift_3.get_drifted_features(return_as_indices=False)


def test_ks_exceptions():

    histograms_src = [np.array([5, 5, 5]), np.array([1, 2, 3])]
    histograms_target = [np.array([1, 1, 1]), np.array([1, 2, 3])]

    # Dataset not specified
    with pytest.raises(Exception):
        ks_drift = KSDrift(features=["f1", "f2"], p_val=0.01, correction="bonferroni")
        drift_metrics = ks_drift.calculate_drift()

    # Histograms specify instead of dataset
    with pytest.raises(Exception):
        ks_drift = KSDrift(distr_src=histograms_src, distr_target=histograms_target,
                           features=["f1", "f2"], p_val=0.01, correction="bonferroni")
        drift_metrics = ks_drift.calculate_drift()

    # invalid dataset type
    with pytest.raises(Exception):
        ks_drift = KSDrift(X_src=[[2, 1], [3, 2]], X_target=[[3, 1], [3, 4]], features=["f1", "f2"], p_val=0.01,
                           correction=None)
        drift_metrics_1 = ks_drift.calculate_drift()

    # Bad shape
    with pytest.raises(Exception):
        ks_drift = KSDrift(X_src=np.zeros((3, 3, 3)), X_target=np.zeros((3, 3, 3)),
                           features=["f1", "f2"], p_val=0.01, correction="bonferroni")
        drift_metrics = ks_drift.calculate_drift()

    # X_src and X_target different number of features
    with pytest.raises(Exception):
        ks_drift = KSDrift(X_src=np.zeros((5, 2)), X_target=np.zeros((5, 3)),
                           features=["f1", "f2"], p_val=0.01, correction="bonferroni")
        drift_metrics = ks_drift.calculate_drift()

    # Invalidad correction method
    with pytest.raises(Exception):
        ks_drift = KSDrift(X_src=np.zeros((5, 2)), X_target=np.zeros((5, 2)), features=["f1", "f2"],
                             p_val=0.01, correction="invalid")
        drift_metrics = ks_drift.calculate_drift()

    # Trying to plot before calculating drift
    with pytest.raises(Exception):
        ks_drift = KSDrift(X_src=np.zeros((3, 3)), X_target=np.zeros((3, 3)), features=["f1", "f2"],
                             p_val=0.01, correction="bonferroni")
        ks_drift.plot_feature_drift_scores()

    # P_value None
    with pytest.raises(Exception):
        ks_drift = KSDrift(X_src=np.zeros((3, 3)), X_target=np.zeros((3, 3)), features=["f1", "f2"], p_val=None)
        ks_drift.calculate_drift()

    # Misuse _get index_feature
    with pytest.raises(Exception):
        ks_drift = KSDrift(X_src=np.zeros((3, 3)), X_target=np.zeros((3, 3)), features=["f1", "f2"])
        ks_drift._get_index_feature(idx_feature=None, name_feature=None)
    with pytest.raises(Exception):
        ks_drift = KSDrift(X_src=np.zeros((3, 3)), X_target=np.zeros((3, 3)), features=["f1", "f2"])
        ks_drift._get_index_feature(idx_feature=3, name_feature="f1")
