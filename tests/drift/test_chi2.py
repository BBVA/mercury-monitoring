import numpy as np
import pytest

from mercury.monitoring.drift.chi2_drift_detector import Chi2Drift


def test_chi2_drift():
    histograms_src = [np.array([5, 5, 5]), np.array([3, 6, 9])]
    histograms_target_1 = [np.array([2, 2, 2]), np.array([1, 2, 3])]
    histograms_target_2 = [np.array([4, 1, 8]), np.array([20, 3, 2])]

    # Case No Drift
    chi2_drift_1 = Chi2Drift(
        distr_src=histograms_src, distr_target=histograms_target_1, features=["f1", "f2"],
        correction="bonferroni", p_val=0.05
    )
    drift_metrics_1 = chi2_drift_1.calculate_drift()
    assert drift_metrics_1["drift_detected"] == False
    assert len(drift_metrics_1["p_vals"]) == 2
    assert drift_metrics_1["threshold"] == (0.05 / len(histograms_src))

    # Case Drift
    chi2_drift_2 = Chi2Drift(
        distr_src=histograms_src, distr_target=histograms_target_2, features=["f1", "f2"],
        correction="bonferroni", p_val=0.05
    )
    drift_metrics_2 = chi2_drift_2.calculate_drift()
    assert drift_metrics_2["drift_detected"] == True
    assert len(drift_metrics_2["p_vals"]) == 2
    assert chi2_drift_2.get_drifted_features() == ['f2']
    assert drift_metrics_2['score'] > drift_metrics_1['score']

    # Plotting doesn't crash
    chi2_drift_2.plot_histograms_drifted_features(figsize=(10, 3))
    chi2_drift_2.plot_feature_drift_scores()

    drift_metrics_2 = chi2_drift_2.calculate_drift()


def test_chi2_drift_exceptions():
    # Histograms are not spceified
    with pytest.raises(Exception):
        chi_drift = Chi2Drift(
            features=["f1", "f2"],
            correction="bonferroni", p_val=0.05
        )
        chi_drift.calculate_drift()

    # Datasets specified instead of histograms
    with pytest.raises(Exception):
        chi_drift = Chi2Drift(X_src=np.zeros((5, 2)), X_target=np.zeros((5, 2)), features=["f1", "f2"],
                              correction="bonferroni", p_val=0.05
                              )
        chi_drift.calculate_drift()

    # Different number of features for source and target histogram
    with pytest.raises(Exception):
        chi_drift = Chi2Drift(
            distr_src=[np.array([5, 5, 5]), np.array([3, 6, 9])],
            distr_target=[np.array([5, 5, 5])],
            features=["f1", "f2"], correction="bonferroni", p_val=0.05
        )
        chi_drift.calculate_drift()

    # Histogram of a feature has a different number of bins
    with pytest.raises(Exception):
        chi_drift = Chi2Drift(
            distr_src=[np.array([5, 5]), np.array([3, 6, 9])],
            distr_target=[np.array([5, 5, 5]), np.array([3, 4, 3])],
            features=["f1", "f2"], correction="bonferroni", p_val=0.05
        )
        chi_drift.calculate_drift()

