import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from typing import List, Tuple, Callable, Union

from mercury.monitoring.drift.ks_drift_detector import KSDrift
from mercury.monitoring.drift.chi2_drift_detector import Chi2Drift
from mercury.monitoring.drift.drift_simulation import BatchDriftGenerator
from mercury.dataschema import DataSchema

import warnings


class PerformancePredictor:

    def __init__(
        self,
        model: "BaseEstimator",  # noqa: F821
        metric_fn: Callable,
        corruptions: List[Tuple] = None,
        percentiles: Union[List, np.array] = None,
        performance_predictor: "BaseEstimator" = None,  # noqa: F821
        param_grid: dict = None,
        K_cv: int = 5,
        random_state: int = None,
        store_train_data: bool = False
    ):
        """
        This class allow us to estimate the performance of a model on an unlabeled dataset, for example to monitor performance
        in production data when we still don't have the labels. The method is based on the paper
        Learning to Validate the Predictions of Black Box Classifiers on Unseen Data. In a nutshell, the steps of the method are:

        1) Apply corruptions to a held-out (labeld) dataset
        2) Obtain percentiles of model outputs and the performance of the model when applying these corruptions
        3) Train a regressor model to predict model performance. The samples to train this regressor model are the percentiles and
            performances obtained in 2)
        4) Use the trained regressor to estimate the performance on serving unlabeled data

        According to the paper, the method works well when: 1) We have a situation of covariate (changes in input data distributions) and
        2) We know in advance what kind of covariate shift we can find in our serving data. However, in our experiments we have found
        that in some situations the method still works when the data also suffers from label shift.
        At the same time, it is important to mention that the method is not 100% accurate and cannot detect performance drop in all cases.

        Original paper:
        https://ssc.io/pdf/mod0077s.pdf

        Args:
            model: The model that we want to estimate the performance
            metric_fn: Function that calculates the metric that we want to estimate. The function should accept the true labels as
                first argument and the predictions as the second argument. For example, you can use functions from sklearn.metrics module.
            corruptions: Optional list of corruptions to apply in the dataset specified in `fit` method.
                If we specify them, we use a list of tuples where each tuple has two elements:

                1) A string with the type of drift to apply.
                2) A dictionary with the parameters of the drift to apply. For the first element you can use any method available in
                mercury.monitoring.drift.drift_simulation.BatchDriftGenerator class. In the second element, the parameters are the
                arguments of the drift function. You can see the tutorial of class or the BatchDriftGenerator documentation for more
                details. If not specified the corruptions will be added in the `fit()` method according to the drift detected.

            percentiles: np.array or list with percentiles to calculate in model outputs to be used as features in the regressor.
                By default, the calculated percentiles are [0, 5, 10, ..., 95, 100]
            performance_predictor: (unfitted) model to use as regressor. By default it will be a RandomForestRegressor with n_estimators=15
            param_grid: dictionary with the hyperparameters grid that will be used when doing a grid search when training the regressor.
                By default just the the max_depth of the RandomForestRegressor is tunned.
            K_cv: Number of folds to use when doing the GridSearch cross-validation to train the regressor. By default 5 will be used
            random_state: random state to use in the RandomForestRegressor. By default is None.
            store_train_data: whether to store the data to train the regressor in the attributes `X_train_regressor` and
                `y_train_regressor`. This can be useful for analysis when performing some experiments of the method. By default is False.

        Example:
            ```python
            >>> model.fit(X_train, y_train)
            >>> from mercury.monitoring.estimation.performance_predictor import PerformancePredictor
            >>> from sklearn.metrics import accuracy_score
            >>> performance_predictor = PerformancePredictor(model, metric_fn=accuracy_score, random_state=42)
            >>> performance_predictor.fit(X=df_test[features], y=df_test[label], X_serving=df_serving[features])
            ```
        """
        self.model = model
        self.metric_fn = metric_fn
        self.corruptions = [] if corruptions is None else corruptions
        self.percentiles = np.arange(0, 101, 5) if percentiles is None else percentiles
        if performance_predictor is None:
            self.performance_predictor_unfitted = RandomForestRegressor(n_estimators=15, criterion='mae', random_state=random_state)
        else:
            self.performance_predictor_unfitted = performance_predictor
        self.param_grid = {'max_depth': np.arange(3, 16, 1), 'criterion': ['absolute_error']} if param_grid is None else param_grid
        self.K_cv = K_cv
        self.store_train_data = store_train_data
        self.performance_predictor = None

    def fit(
        self,
        X: "pandas.DataFrame",  # noqa: F821
        y: Union["pandas.DataFrame", "np.array"],  # noqa: F821
        dataset_schema: "mercury.dataschema.DataSchema" = None,  # noqa: F821
        names_categorical: list = None,
        X_serving: "pandas.DataFrame" = None  # noqa: F821
    ):
        """
        Fits the regressor to predict the performance using a dataset not used as training data.

        Args:
            X: Pandas dataframe with the inputs of our model. It should be a held-out dataset not used to train the model
            y: corresponding labels of `X`
            dataset_schema: a DataSchema object. If not passed, it is created automatically
            names_categorical: list of categorical columns. Only used if `dataset_schema` is not specified. In that case, it
                will take this list as categorical columns
            X_serving: optional dataframe with the serving data (without labels). If specified, it will detect drift between
                `X` and `X_serving` and the corruptions will be added based on that drift.
        """

        # Generate Schema
        self._generate_schema(X, dataset_schema, names_categorical)

        # if X_serving is passed, then add new error generators based on data drift
        if X_serving is not None:
            self.corruptions.extend(self._create_corruptions_from_data_drift(X, X_serving))

        # if we have very few corruptions (less than param K_cv), raise a warning a create scale drift
        if len(self.corruptions) <= self.K_cv:
            warnings.warn(
                "Very corruptions have been specified or created from data drift. "
                "scale dirft will be added for all features individually")
            continous_feats = self.dataset_schema.continuous_feats + self.dataset_schema.discrete_feats
            for f in continous_feats:
                self.corruptions.extend(self._create_scale_drift(feature=f))

        X_train_regressor = []
        y_train_regressor = []
        for corruption in self.corruptions:

            corruption_fn = corruption[0]
            corruption_args = corruption[1]
            # Apply drift generator
            X_corrupt = self._apply_corruption(X, corruption_fn, corruption_args)

            # Score on corrupted examples
            y_hat_corrupt = self.model.predict(X_corrupt)
            score_corrupt = self.metric_fn(y, y_hat_corrupt)

            # Statitics of model outputs (percentiles)
            statistics_outputs = self._get_statistics_model_outputs(X_corrupt)

            # Add data point to samples for performance predictor
            X_train_regressor.append(statistics_outputs)
            y_train_regressor.append(score_corrupt)

        # Train performance predictor regressor
        self._fit_performance_predictor(X_train_regressor, y_train_regressor)

        # Store performance predictor trai data if specified
        if self.store_train_data:
            self.X_train_regressor = X_train_regressor
            self.y_train_regressor = y_train_regressor

        return self

    def _generate_schema(self, X, dataset_schema, names_categorical=None):
        """
        Generates the dataset schema if not specified and stores it `dataset_schema` attirbute
        """
        if dataset_schema is not None:
            self.dataset_schema = dataset_schema
        elif names_categorical is not None:
            self.dataset_schema = DataSchema().generate_manual(
                dataframe=X,
                categ_columns=names_categorical,
                discrete_columns=[],
                binary_columns=[]
            )
        else:
            self.dataset_schema = DataSchema().generate(X)

    def _create_corruptions_from_data_drift(self, X_source, X_target):
        """
        Creates corruptions by detecting drift between `X_source` and `X_target`
        """

        corruptions = []

        # Numerical Features: Get Features with drift with KSDrift
        continous_feats = self.dataset_schema.continuous_feats + self.dataset_schema.discrete_feats
        if len(continous_feats) > 0:
            ks_drift = KSDrift(X_src=X_source[continous_feats].values, X_target=X_target[continous_feats].values, features=continous_feats)
            drift_result = ks_drift.calculate_drift()
            # Apply different kinds of drift for continuos features
            for feat in ks_drift.get_drifted_features():

                # Shift Drift
                corruptions.extend(self._create_shift_drift(X_source, X_target, feat))

                # Scale Drift
                corruptions.extend(self._create_scale_drift(feat))

                # Outliers Drift
                corruptions.extend(self._create_outliers_drift(feat))

            # Hyperplane Rotation Drift
            if len(ks_drift.get_drifted_features()) > 0:
                corruptions.extend(self._create_hyperplane_rotation_drift(ks_drift.get_drifted_features()))

        # Categorical Features: Chi-Square Drift
        cat_feats = self.dataset_schema.binary_feats + self.dataset_schema.categorical_feats
        if len(cat_feats) > 0:
            src_histograms, tgt_histograms = _get_histogram_categoricals(
                X_source[cat_feats], X_target[cat_feats], cat_feats
            )
            chi2_drift = Chi2Drift(
                distr_src=src_histograms,
                distr_target=tgt_histograms,
                features=cat_feats
            )
            drift_result = chi2_drift.calculate_drift()
            for feat in chi2_drift.get_drifted_features():

                # Recodification drift
                for i in range(min(X_source[feat].nunique(), 10)):
                    corruptions.append(('recodification_drift', {'cols': [feat]}))

        return corruptions

    def _create_shift_drift(self, X_source, X_target, feature):
        """
        Returns list with shift drift specifications
        """
        shift_drift_corruptions = []

        diff_q95 = X_target[feature].quantile(0.95) - X_source[feature].median()
        diff_q05 = X_target[feature].quantile(0.05) - X_source[feature].median()
        forces_neg = np.linspace(diff_q05, 0, num=10)
        forces_pos = np.linspace(0, diff_q95 , num=10)
        forces = list(forces_neg) + list(forces_pos)
        noises = [X_source[feature].std()] * len(forces)
        for force, noise in zip(forces, noises):
            drift_args = {
                'cols': [feature],
                'force': force,
                'noise': noise
            }
            shift_drift_corruptions.append(('shift_drift', drift_args))

        return shift_drift_corruptions

    def _create_scale_drift(self, feature):
        """
        Returns list with scale drift specifications
        """
        scale_drift_corruptions = []
        forces = [0.1, 0.5, 0.8, 1.2, 1.5, 2, 2.5, 3, 5, 10, 20, 100]
        for f in forces:
            drift_args = {
                'cols': [feature],
                'mean': f
            }
            scale_drift_corruptions.append(('scale_drift', drift_args))
        return scale_drift_corruptions

    def _create_outliers_drift(self, feature):
        """
        Returns list with outliers drift specifications
        """
        outliers_drift_corruptions = []
        for perc in [0.05, 0.95]:
            for proportion in [0.25, 0.5, 0.75]:
                drift_args = {
                    'cols': [feature],
                    'method': 'percentile',
                    'method_params': {
                        'percentile': perc,
                        'proportion': proportion
                    }
                }
                outliers_drift_corruptions.append(('outliers_drift', drift_args))
        return outliers_drift_corruptions

    def _create_hyperplane_rotation_drift(self, features, num_drifts=20):
        """
        Returns list with hyperplane drift specifications
        """
        hyperplane_drift_corruptions = []
        for force in np.linspace(0, 90, num=num_drifts):
            drift_args = {
                'cols': features,
                'force': force
            }
            hyperplane_drift_corruptions.append(('hyperplane_rotation_drift', drift_args))
        return hyperplane_drift_corruptions

    def _apply_corruption(self, X, corruption_fn, corruption_args):
        """
        apply corruption `corruption_fn` using `corruption_args` arguments to `X`
        """

        corruption_generator = BatchDriftGenerator(X=X.copy(), schema=self.dataset_schema)
        corruption_gen_fun = getattr(corruption_generator, corruption_fn, None)
        if not callable(corruption_gen_fun):
            raise RuntimeError(
                "corruption_fn = %s must be a method of BatchDriftGenerator."
            )
        corrupted_gen = corruption_gen_fun(**corruption_args)
        return corrupted_gen.data

    def _get_statistics_model_outputs(self, X):
        """
        Obtains percentiles of model outputs
        """

        try:
            # Try predict_proba.
            y_hat = self.model.predict_proba(X)
        except AttributeError:
            # If predict_proba is not defined in model, do predict (eg. for regression and tf models)
            y_hat = self.model.predict(X)

        if len(y_hat.shape) == 2:
            # classification case
            if y_hat.shape[1] == 2:
                # Binary Classification. Just take percentiles of positive class
                statistics = np.percentile(y_hat[:, 1], q=self.percentiles)
            else:
                # Compute Percentiles for all classes
                l_percentiles = []
                for i in range(y_hat.shape[1]):
                    l_percentiles.append(
                        np.percentile(y_hat[:, i], q=np.arange(0, 101, 5))
                    )
                statistics = np.array(l_percentiles).flatten()
        else:
            statistics = np.percentile(y_hat, q=self.percentiles)

        return statistics

    def _fit_performance_predictor(self, X, y):
        """
        Fit the performance predictor using the GridSearchCV
        """

        self.performance_predictor = GridSearchCV(
            self.performance_predictor_unfitted,
            param_grid=self.param_grid,
            cv=self.K_cv,
            scoring='neg_mean_absolute_error')\
            .fit(X, y)

    def predict(self, X_serving):
        """
        Returns the estimated performance on `X_serving`
        """

        # Statistics Model Outputs
        statistics_outputs = self._get_statistics_model_outputs(X_serving)

        # Predict Score on Serving Data
        predicted_score = self.performance_predictor.predict(statistics_outputs.reshape(1, -1))
        return predicted_score


def _get_histogram_categorical_column(X_source, X_target, cat_feat):

    counts_source = X_source[cat_feat].value_counts()
    counts_target = X_target[cat_feat].value_counts()

    domain = set(counts_source.index).union(set(counts_target.index))

    source_hist = np.zeros(len(domain))
    target_hist = np.zeros(len(domain))

    for i, value in enumerate(domain):
        if value in counts_source:
            source_hist[i] = counts_source[value]
        if value in counts_target:
            target_hist[i] = counts_target[value]

    return source_hist, target_hist


def _get_histogram_categoricals(X_source, X_target, cat_feats):
    src_histograms = []
    tgt_histograms = []
    for feat in cat_feats:
        src_histogram, tgt_histogram = _get_histogram_categorical_column(X_source, X_target, feat)
        src_histograms.append(src_histogram)
        tgt_histograms.append(tgt_histogram)
    return src_histograms, tgt_histograms