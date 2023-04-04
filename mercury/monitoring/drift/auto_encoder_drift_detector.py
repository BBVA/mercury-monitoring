import numpy as np

from scipy.stats import mannwhitneyu
from sklearn.model_selection import train_test_split

from typing import List, Callable, Union, Any


class AutoEncoderDriftDetector():
    """
    This class uses several autoencoders for testing whether the joint distributions of two
    datasets are the same. In order to do this, it makes bootstrap samples from a source
    dataset (X_src), trains an autoencoder on each one of them (using training/test splits) and
    it stores the distribution of reconstruction errors (i.e. the error distribution reconstructing the
    source dataset). The default reconstruction error is the MSE, although you can specify your own.

    Then, when target data is available (X_target), it calculates the error distribution on the new
    environment and compares it with the one obtained on X_src by using a Mann-Withney-U statistical test
    under the null hypothesis (H0) that no drift exists (that is, the two error samples come from the same
    distribution).

    Args:
        p_val (float):
            Significance level for rejecting the null hypothesis. Default value is 0.05.
        bootstrap_samples (int):
            How many bootstrap samples to make
        bootstrap_sample_size (int):
            The size of each bootstrap sample.
        autoencoder_kwargs (dict):
            Dictionary with arguments for training the internal `n` autoencoders.
        custom_reconstruction_error (callable or string):
            Custom function for measuring reconstruction quality. This is only used for the
            drift calculation, not for training the autoencoders (use autoencoder_kwargs for that). This
            can be a string in (mse, rmse, mae), meaning (Mean Squared Error, Root Mean Squared Error,
            Mean Absolute Loss error). Alternatively, you can also provide a custom function with the form
            `custom_fn(ytrue, ypred)` that returns the reconstruction error without reductions. That is,
            a matrix with the same shape as ytrue/ypred. Default is Mean squared error.
        fitted_models (list):
            List with trained autoencoders (tf.keras.Model). Use it only if you want to "reload" an
            already trained detector.
        reference_errors:
            List with the errors obtained at the time of training the detector. Use it only
            if you want to "reload" an already trained detector.

    Example:
        ```python
        >>> from mercury.monitoring.drift.auto_encoder_drift_detector import AutoEncoderDriftDetector
        # Create example datasets
        >>> X_src = np.array([np.random.normal(0, 1, 1000), np.random.normal(0, 1, 1000)]).T
        >>> X_target = np.array([np.random.normal(0, 1, 500) + 2, np.random.normal(0, 1, 500)]).T
        # Train the detector on the source dataset
        >>> drift_detector = AutoEncoderDriftDetector()
        >>> drift_detector = drift_detector.fit(X_src)
        # Use the detector for checking whether drift exists on X_target
        >>> drift_detector.calculate_drift(X_target)
        "{'pval': 7.645836962858183e-06, 'drift': True}"
        ```
    """
    def __init__(self,
                 p_val: float = 0.05,
                 bootstrap_samples: int = 30,
                 bootstrap_sample_size: int = 10_000,
                 autoencoder_kwargs: dict = None,
                 custom_reconstruction_error: Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = 'mse',
                 fitted_models: list = None,
                 reference_errors: List[float] = None):

        self.bootstrap_samples = bootstrap_samples
        self.bootstrap_sample_size = bootstrap_sample_size
        self.pval = p_val
        self.ae_args = autoencoder_kwargs if autoencoder_kwargs else dict()
        self.reconstruction_metric_fn = self._get_reconstruction_err_fn(custom_reconstruction_error)

        # Support case when user "reloads" the detector
        if fitted_models is not None:
            if reference_errors is None or len(reference_errors) != len(fitted_models):
                raise ValueError("""If you pass a list of recovered fitted models you must also provide a list of """
                                 """reference_errors of the same length as `fitted_models`.""")
            self.fitted_models = fitted_models
            self.reference_distribution = reference_errors
        if reference_errors is not None and fitted_models is None:
            raise ValueError("For reloading the detector you must pass both `reference_errors` and `fitted_models`.")

        if not isinstance(self.ae_args, dict):
            raise ValueError("Error. autoencoer_kwargs must be a dictionary.")

        self._init_autoencoder_params()

    def _get_reconstruction_err_fn(self, function: Any) -> Callable:
        fn = lambda y_true, y_pred: (y_true - y_pred)**2  # noqa: E731

        if function == 'mae':
            fn = lambda y_true, y_pred: np.abs(y_true - y_pred)  # noqa: E731

        if callable(function):
            fn = function

        return fn

    def _init_autoencoder_params(self):
        """Sets default values for the AE training parameters (if not already provided)."""
        if 'optimizer' not in self.ae_args:
            self.ae_args['optimizer'] = 'adam'
        if 'loss' not in self.ae_args:
            self.ae_args['loss'] = 'mse'
        if 'batch_size' not in self.ae_args:
            self.ae_args['batch_size'] = 128
        if 'epochs' not in self.ae_args:
            self.ae_args['epochs'] = 10
        if 'verbose' not in self.ae_args:
            self.ae_args['verbose'] = False

    def _build_autoencoder(self, n_inputs):
        """
        This function builds a simple autoencoder model when one is not provided.

        Args:
            n_inputs (int): number of features passed into the model.

        Returns:
            (tf.keras.Model): autoencoder model
        """
        import tensorflow as tf  # Lazy import in order to avoid loading tf when importing this module.

        n_hidden = 2 if n_inputs < 100 else 8

        inputs = tf.keras.Input(shape=(n_inputs,))
        encoder = tf.keras.layers.Dense(n_inputs)(inputs)
        embedding = tf.keras.layers.Dense(n_hidden)(encoder)
        decoder = tf.keras.layers.Dense(n_inputs)(embedding)

        model = tf.keras.Model(inputs=inputs, outputs=decoder)
        return model

    def _subsample(self, matrix: np.ndarray, n=10_000):
        """Subsamples the rows of a given matrix with replacement."""
        indices = np.random.choice(matrix.shape[0], size=n)
        return matrix[indices, :]

    def _get_data_splits(self, dataset: np.ndarray, subsample=10_000):
        """Takes a dataset, resamples it and returns training/test splits."""
        dataset = self._subsample(dataset, n=subsample)
        train, test = train_test_split(dataset, test_size=.4)
        return train, test

    def _get_simulation_error_distr(self, dataset: np.ndarray):
        metrics_no_drift = []
        per_feature_metrics = []
        models = []
        for _ in range(self.bootstrap_samples):
            source0, source1 = self._get_data_splits(dataset, subsample=self.bootstrap_sample_size)

            model = self._build_autoencoder(source0.shape[-1])
            model.compile(optimizer=self.ae_args['optimizer'], loss=self.ae_args['loss'])
            model.fit(source0, source0,
                      epochs=self.ae_args['epochs'],
                      batch_size=self.ae_args['batch_size'],
                      verbose=self.ae_args['verbose'])

            y_preds = model.predict(source1)
            temp = self.reconstruction_metric_fn(source1, y_preds)
            mse = np.mean(temp)
            mse_per_feat = np.mean(temp, axis=0)

            metrics_no_drift.append(mse)
            per_feature_metrics.append(mse_per_feat)

            models.append(model)

        self.fitted_models = models
        self.mse_reference = {'mses': metrics_no_drift, 'feature_mses': per_feature_metrics}
        return metrics_no_drift

    def _get_real_error_distr(self, dataset: np.ndarray):
        """ Gets the errors on a given dataset using all the trained autoencoders. """
        metrics_drift = []
        per_feature_metrics = []

        for _, model in enumerate(self.fitted_models):
            train_ds = self._subsample(dataset, n=self.bootstrap_sample_size)
            y_preds = model.predict(train_ds)
            temp = self.reconstruction_metric_fn(train_ds, y_preds)
            mse = np.mean(temp)
            mse_per_feat = np.mean(temp, axis=0)

            metrics_drift.append(mse)
            per_feature_metrics.append(mse_per_feat)

        self.mse_target = {'mses': metrics_drift, 'feature_mses': per_feature_metrics}
        return metrics_drift

    @property
    def is_fitted(self) -> bool:
        """ Whether the detector has been fitted and it's ready for calculating drift """
        return hasattr(self, 'reference_distribution') and hasattr(self, 'fitted_models')

    def fit(self, X_src: np.ndarray) -> "AutoEncoderDriftDetector":
        """ Trains the detector using a source (reference) dataset.

        Args:
            X_src: the reference dataset.

        Returns:
            the detector itself.
        """
        self.reference_distribution = self._get_simulation_error_distr(X_src)
        return self

    def calculate_drift(self, X: np.ndarray, return_errors: bool = False) -> dict:
        """ Checks whether drift exists on a given dataset using X_src as reference:

        Args:
            X: the target dataset.
            return_errors: Whether to include the reconstruction errors in the result.

        Returns:
            Dictionary containing the result of the test and, optionally, the reference
            and target reconstruction errors.

        """
        if not self.is_fitted:
            raise RuntimeError("Error. This detector han't been trained yet. Call `fit` first.")

        self.target_distribution = self._get_real_error_distr(X)

        test = mannwhitneyu(self.mse_reference['mses'], self.mse_target['mses'], alternative='less')

        result = {
            'pval': test.pvalue,
            'drift': test.pvalue < self.pval
        }

        if return_errors:
            result['source_errors'] = self.mse_reference['mses']
            result['target_errors'] = self.mse_target['mses']

        return result