import numpy as np
import pytest

from mercury.monitoring.drift.histogram_distance_drift_detector import HistogramDistanceDrift


def test_histogram_distance_drift():
    histograms_src = [np.array([5, 5, 5]), np.array([3, 6, 9])]
    histograms_target_1 = [np.array([2, 2, 2]), np.array([1, 2, 3])]
    histograms_target_2 = [np.array([4, 1, 8]), np.array([30, 3, 2])]

    for distance_metric in ["hellinger", "jeffreys", "psi"]:
        # Case No Drift
        hist_dist_drift_1 = HistogramDistanceDrift(
            distr_src=histograms_src, distr_target=histograms_target_1, features=["f1", "f2"],
            distance_metric=distance_metric, correction="bonferroni", p_val=0.001, n_permutations=50
        )
        drift_metrics_1 = hist_dist_drift_1.calculate_drift()
        assert drift_metrics_1["drift_detected"] == False
        assert len(drift_metrics_1["p_vals"]) == 2
        assert drift_metrics_1["threshold"] == (0.001 / len(histograms_src))

        # Case Drift
        hist_dist_drift_2 = HistogramDistanceDrift(
            distr_src=histograms_src, distr_target=histograms_target_2, features=["f1", "f2"],
            distance_metric=distance_metric, correction="bonferroni", p_val=0.05, n_permutations=50
        )
        drift_metrics_2 = hist_dist_drift_2.calculate_drift()
        assert drift_metrics_2["drift_detected"] == True
        assert len(drift_metrics_2["p_vals"]) == 2
        assert 'f2' in hist_dist_drift_2.get_drifted_features()
        assert drift_metrics_2['score'] > drift_metrics_1['score']

        # Plotting doesn't crash
        hist_dist_drift_2.plot_histograms_drifted_features(figsize=(10, 3))
        hist_dist_drift_2.plot_feature_drift_scores()

        # Set back the first target histogram
        hist_dist_drift_2.set_distributions(histograms_src, histograms_target_1)
        hist_dist_drift_2.p_val = 0.001
        assert hist_dist_drift_2.distr_target == histograms_target_1
        assert hist_dist_drift_2.drift_metrics is None
        drift_metrics = hist_dist_drift_2.calculate_drift()
        assert drift_metrics["drift_detected"] == False

        # Test when features are not specified:
        hist_dist_drift_3 = HistogramDistanceDrift(
            distr_src=histograms_src, distr_target=histograms_target_2,
            distance_metric=distance_metric, correction="bonferroni", p_val=0.05, n_permutations=50
        )
        drift_metrics_3 = hist_dist_drift_3.calculate_drift()
        # Plotting doesn't crash
        hist_dist_drift_3.plot_histograms_drifted_features()
        hist_dist_drift_3.plot_feature_drift_scores(top_k=1)
        assert hist_dist_drift_3.get_drifted_features(return_as_indices=True) == [1]
        with pytest.raises(Exception):
            hist_dist_drift_3.get_drifted_features(return_as_indices=False)


def test_histogram_distance_drift_exceptions():
    # Histograms are not spceified
    with pytest.raises(Exception):
        hist_dist_drift = HistogramDistanceDrift(
            features=["f1", "f2"],
            distance_metric="hellinger", correction="bonferroni", p_val=0.05, n_permutations=50
        )
        hist_dist_drift.calculate_drift()

    # Datasets specified instead of histograms
    with pytest.raises(Exception):
        hist_dist_drift = HistogramDistanceDrift(X_src=np.zeros((5, 2)), X_target=np.zeros((5, 2)),
                                                 features=["f1", "f2"],
                                                 distance_metric="hellinger", correction="bonferroni", p_val=0.05,
                                                 n_permutations=50
                                                 )
        hist_dist_drift.calculate_drift()

    # Different number of features for source and target histogram
    with pytest.raises(Exception):
        hist_dist_drift = HistogramDistanceDrift(
            distr_src=[np.array([5, 5, 5]), np.array([3, 6, 9])],
            distr_target=[np.array([5, 5, 5])],
            features=["f1", "f2"], distance_metric="hellinger", correction="bonferroni", p_val=0.05, n_permutations=50
        )
        hist_dist_drift.calculate_drift()

    # Histogram of a feature has a different number of bins
    with pytest.raises(Exception):
        hist_dist_drift = HistogramDistanceDrift(
            distr_src=[np.array([5, 5]), np.array([3, 6, 9])],
            distr_target=[np.array([5, 5, 5]), np.array([3, 4, 3])],
            features=["f1", "f2"], distance_metric="hellinger", correction="bonferroni", p_val=0.05, n_permutations=50
        )
        hist_dist_drift.calculate_drift()

    # unexistent distance metrics
    with pytest.raises(Exception):
        hist_dist_drift = HistogramDistanceDrift(
            distr_src=[np.array([5,5,5]), np.array([3, 6, 9])],
            distr_target=[np.array([5,5,5]), np.array([3, 4, 3])],
            features=["f1", "f2"], distance_metric="distance_1", correction="bonferroni", p_val=0.05,
            n_permutations=50
        )
        hist_dist_drift.calculate_drift()

def test_histogram_with_empty_bins():
    histograms_src = [np.array([5, 1, 0]), np.array([0, 6, 0])]
    histograms_target_1 = [np.array([2, 0, 1]), np.array([1, 0, 3])]

    hist_dist_drift_1 = HistogramDistanceDrift(
        distr_src=histograms_src, distr_target=histograms_target_1, features=["f1", "f2"],
        distance_metric="hellinger", correction="bonferroni", p_val=0.05, n_permutations=50
    )
    drift_metrics_1 = hist_dist_drift_1.calculate_drift()
    assert len(drift_metrics_1["p_vals"]) == 2