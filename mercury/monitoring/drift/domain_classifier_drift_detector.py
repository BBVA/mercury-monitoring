import numpy as np
import pandas as pd

from scipy.stats import binom_test
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from .base import BaseBatchDriftDetector


class DomainClassifierDrift(BaseBatchDriftDetector):
    """
    This class is a drift detector that allows us to train a classifier (Random Forest) to distinguish between a
    source dataset `X_src` and a target dataset `X_target`. The source and target datasets are specified when creating
    the object together with other available parameters. Then the method `calculate_drift()` is used to obtain a dict
    of metrics.

    Args:
        X_src (np.array):
            The source dataset.
        X_target (np.array):
            The target dataset.
        features (list):
            The name of the features of `X_src` and `X_target`.
        p_val (float):
            Threshold to be used with the p-value obtained to decide if there is drift or not (affects
            `drift_detected` metric when calling `calculate_drift()`)
            Lower values will be more conservative when indicating that drift exists in the dataset.
            Default value is 0.05.
        test_size (float):
            When training the domain classifier, this parameter indicates the size of the test set of the domain
            classifier. Since the metric `drift_score` is the auc of this test size, the drift_score won't be very
            accurate if the test_size is too small. Default value is 0.3
        n_runs (int):
            This is the number of times that the domain classifier is trained with different splits. The returned
            metrics correspond to the averages of different runs. A higher number of runs will produce more accurate
            metrics, but it will be computationally more expensive.
        alpha (float):
            This parameter impacts directly to the false drift detection rate. Concretely, it specifies how much better
            the AUC of the domain classifier has to be than 0.5 (as a random classifier) to consider that successfully
            distinguishes samples from the source dataset and target dataset. Increasing the value will be more
            conservative when detecting drift, so it will lower the false positive detection rate. However, higher
            values also can miss true drift detection. Default value is 0.025 which in general a good trade-off.
        **kwargs:
            The rest of the parameters will be used in the Random forest trained as domain classifier. If not specified,
            then the default parameters will be used.

    Example:
        ```python
        >>> from mercury.monitoring.drift.domain_classifier_drift import DomainClassifierDrift
        # Create example datasets
        >>> X_src = np.array([np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)]).T
        >>> X_target = np.array([np.random.normal(0, 1, 500) + 2, np.random.normal(0, 1, 500)]).T
        # Calculate drift
        >>> drift_detector = DomainClassifierDrift(
        ...     X_src, X_target, features=["f1", "f2"], p_val=0.01, test_size=0.3, n_runs=10)
        >>> drift_metrics = drift_detector.calculate_drift()
        # Print the metrics
        >>> print("Drift Score: ", drift_metrics["score"])
        >>> print("p_val: ", drift_metrics["p_val"])
        >>> print("Is drift detected? ", drift_metrics["drift_detected"])
        # Get drift scores per feature
        >>> drift_detector.plot_feature_drift_scores(figsize=(8,4))
        ```
    """
    def __init__(self, X_src=None, X_target=None, features=None, p_val=0.05, test_size=0.3, n_runs=10,
                 alpha=0.025, **kwargs):
        super().__init__(X_src=X_src, X_target=X_target, features=features, p_val=p_val)
        self.test_size = test_size
        self.n_runs = n_runs
        self.alpha = alpha
        self.kwargs = kwargs

    def _train_domain_classifier(self, test_size=0.3, random_state=None):
        """
        Trains a Random Forest model to discriminate between source and target datasets
        """

        self._check_datasets()

        metrics = {}

        # Split Source dataset (original training set) in train and test
        X_src_train, X_src_test = train_test_split(self.X_src, test_size=test_size, random_state=random_state)
        # Instances of source dataset we set the label 0, indicating that they belong to the source dataset
        y_src_train = np.zeros(X_src_train.shape[0])
        y_src_test = np.zeros(X_src_test.shape[0])

        # Split Target dataset (the new data) in train and test
        X_target_train, X_target_test = train_test_split(self.X_target, test_size=test_size, random_state=random_state)
        # Instances of target dataset we set the label 1, indicating that they belong to the target dataset
        y_target_train = np.ones(X_target_train.shape[0])
        y_target_test = np.ones(X_target_test.shape[0])

        # Concatenate data from both sets
        X_train = np.concatenate([X_src_train, X_target_train])
        y_train = np.concatenate([y_src_train, y_target_train])

        X_test = np.concatenate([X_src_test, X_target_test])
        y_test = np.concatenate([y_src_test, y_target_test])

        # Train Random Forest
        model = RandomForestClassifier(**self.kwargs)
        model = model.fit(X_train, y_train)

        # Get auc score to get how good the classifier can discriminate between both sets
        metrics['auc'] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

        # Get importances, they can tell us which features can be causing the drift if there is
        metrics['importances'] = model.feature_importances_

        return metrics, model

    def _aggregate_domain_classifier_metrics(self, metrics):
        """
        Computes the mean auc of several runs and a p_value based on a binomial test. The mean of importances
        from different runs are also computed.
        """

        agg_metrics = {}

        # Compute p_value with binomial test and average AUC
        l_auc = [m['auc'] for m in metrics]
        n_greater = [auc > (0.5 + self.alpha) for auc in l_auc]
        n_success = np.sum(n_greater)
        p_val = binom_test(n_success, n=len(l_auc), p=0.5, alternative='greater')
        agg_metrics['p_val'] = p_val
        agg_metrics['score'] = np.mean(l_auc)

        # Computes the mean of importances for each feature
        n_features = len(metrics[0]['importances'])
        importances = np.zeros((n_features, len(metrics)))
        for i in range(n_features):
            for j in range(len(metrics)):
                importances[i, j] = metrics[j]['importances'][i]
        agg_metrics['scores'] = np.mean(importances, axis=1)

        return agg_metrics

    def calculate_drift(self, return_drift_score_target=False):
        """
        Calculates the drift metrics. The returned drift metrics are in a dictionary with the next keys:

        - drift_detected: indicates if drift is detected or not, comparing the indicated `p_val` and the obtained
            p-value. The obtained p-value is based on a binomial test that takes into account the number of times
            that the AUC of the domain classifier was higher than random guesses.
        - drift_score: this is the average AUC obtained by the domain classifier over the different runs.

        If the parameter `return_drift_score_target` is True, then it also returns a dataframe with a drift score for
        each sample in the target dataset. This score is the predicted probability by the domain classifier and can
        help to detect the drifted samples.

        Args:
            return_drift_score_target (boolean):
                Indicates if additionally return a dataframe with a drift score for each sample in the target dataset

        Returns:
            (dict): Dictionary with the drift metrics. If parameter `return_drift_score_target` is set to True, then it also
            returns a dataframe with the drift scores for the target dataset.
        """

        metrics_domain_classifier = []
        X_target_hat = []
        for _ in range(self.n_runs):
            m_domain_classifier, model = self._train_domain_classifier(test_size=self.test_size)
            metrics_domain_classifier.append(m_domain_classifier)
            if return_drift_score_target:
                X_target_hat.append(model.predict_proba(self.X_target)[:, 1])

        self.drift_metrics = self._aggregate_domain_classifier_metrics(metrics_domain_classifier)
        self.drift_metrics["drift_detected"] = self.drift_metrics["p_val"] < self.p_val
        if return_drift_score_target:
            df_drift_scores = pd.DataFrame(self.X_target, columns=self.features)
            df_drift_scores["drift_score"] = np.mean(X_target_hat, axis=0)
            return self.drift_metrics, df_drift_scores
        else:
            return self.drift_metrics