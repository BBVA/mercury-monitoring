import numpy as np

from scipy.stats import ks_2samp

from .base import BaseBatchDriftDetector


class KSDrift(BaseBatchDriftDetector):
    """
    This class is a drift detector that allows us to detect drift in a source dataset `X_src` and a target dataset
    `X_target` by calculating the Kolmogorov-Smirnov (KS) statistic. For each feature in the datasets, a
    Kolmogorov-Smirnov test is performed, which calculates the distance between the feature using the samples of each
    dataset. A p-value is also obtained when performing the tests, which are used to decide if exists drift in the
    datasets. The source and target datasets are specified when creating the object together with other available
    parameters. Then the method `calculate_drift()` is used to obtain a dict of metrics.

    Args:
        X_src (np.array):
            The source dataset.
        X_target (np.array):
            The target dataset.
        features (list):
            The name of the features of `X_src` and `X_target`.
        p_val (float):
            Threshold to be used with the p-value obtained to decide if there is drift or not (affects
            `drift_detected` metric when calling `calculate_drift()`).
            Lower values will be more conservative when indicating that drift exists in the dataset.
            Default value is 0.05.
        correction (string):
            Since multiple tests are performed, this parameter controls whether to perform a correction of the p-value
            or not (affects `drift_detected` metric when calling `calculate_drift()`). If None, it doesn't perform
            a correction
            Default value is "bonferroni", which makes the bonferroni correction taking into account the number of tests

    Example:
        ```python
        >>> from mercury.monitoring.drift.ks_drift_detector import KSDrift
        # Create example datasets
        >>> X_src = np.array([np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)]).T
        >>> X_target = np.array([np.random.normal(0, 1, 500) + 2, np.random.normal(0, 1, 500)]).T
        # Calculate drift
        >>> drift_detector = KSDrift(X_src, X_target, features=["f1", "f2"], p_val=0.01, correction="bonferroni")
        >>> drift_metrics = drift_detector.calculate_drift()
        # Print the metrics
        >>> print("Drift Score: ", drift_metrics["score"])
        >>> print("Is drift detected? ", drift_metrics["drift_detected"])
        # Plot feature drift scoress
        >>> ax = drift_detector.plot_feature_drift_scores(top_k=7)
        # Get drifted features
        >>> print(drift_detector.get_drifted_features())
        # Plot distributions of drifted features
        >>> drift_detector.plot_distribution_drifted_features()
        ```
    """

    def __init__(self, X_src=None, X_target=None, features=None, correction="bonferroni", p_val=0.05):
        super().__init__(X_src=X_src, X_target=X_target, features=features, correction=correction, p_val=p_val)

    def _calculate_ks(self):
        """
        Computes Drift using the Kolmogorov-Smirnov KS Test
        """

        self._check_datasets()

        n_features = self.X_src.shape[1]
        p_val = np.zeros(n_features, dtype=np.float32)
        dist = np.zeros_like(p_val)
        for f in range(n_features):
            dist[f], p_val[f] = ks_2samp(self.X_src[:, f], self.X_target[:, f], alternative='two-sided', mode='asymp')

        return p_val, dist

    def calculate_drift(self):
        """
        Computes Drift using the Kolmogorov-Smirnov KS Test, which calculates the largest difference of the cumulative
        density functions of two variables.
        It performs the test individually for each feature and obtains the p-values and the statistics for each feature.
        The returned dictionary contains the next metrics:

        - p_vals (np.array): the p-values returned by the KS test for each feature.
        - scores (np.array): the distances (ks statistic) return by the KS test for each feature.
        - score: average of all the distances.
        - drift_detected: final decision if drift is detected. If any of the p-values of the features is lower than
            the threshold, then this is set to 1. Otherwise is 0. This will depend on the specified `p_val` and
            `correction` values.
        - threshold: final threshold applied to decide if drift is detected. It depends on the specified
            `p_val` and `correction` values.

        Returns:
            (dict): Dictionary with the drift metrics as specified.

        """

        p_vals_test, distances = self._calculate_ks()
        drift_pred, threshold = self._apply_correction(p_vals_test)
        self.drift_metrics = {
            "p_vals": p_vals_test,
            "scores": distances,
            "score": np.mean(distances),
            "drift_detected": drift_pred,
            "threshold": threshold
        }
        return self.drift_metrics
