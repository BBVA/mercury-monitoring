import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde

from typing import Union, Optional


class DensityDriftDetector:
    """
    This detector calculates the "probability" of a particular sample of being an anomaly (or
    presenting drift) with respect to a source reference dataset.

    This detector works using a VAE for building embeddings of the input data. Then, with those
    source embeddings, the density is estimated in such a way that zones in the embeddings space
    with lots of samples have high density whereas zones with low density represent "less"
    probable samples.

    After having trained the detector it can be used for checking whether a particular (new) sample
    is an anomaly.

    Args:
        embedding_dim (int): embedding dimensionality for the VAE (if no `vae` provided)
        vae: Optional VAE architecture
        encoder: encoder part from `vae`
        decoder: decoder part from `vae`

    Example:
        ```python
        >>> from mercury.monitoring.drift.density_drift_detector import DensityDriftDetector
        >>> detector = DensityDriftDetector().fit(X_source)
        ```
    """

    def __init__(
        self,
        embedding_dim: int = 2,
        vae: "tf.keras.Model" = None,  # noqa: F821
        encoder: "tf.keras.Model" = None,  # noqa: F821
        decoder: "tf.keras.Model" = None  # noqa: F821
    ):

        if encoder or decoder or vae and all(i is None for i in [vae, encoder, decoder]):
            raise ValueError("All of vae, encoder and decoder must be defined.")

        self.vae, self.encoder, self.decoder = vae, encoder, decoder
        self.embedding_dim = encoder.outputs[0].shape[-1] if self.vae else embedding_dim

    def fit(self, source: Union["np.ndarray", "pd.DataFrame"],
            epochs: int = None, batch_size: int = 128):
        """
        Fits the model

        Args:
            source: Reference dataset
            epochs: Number of epochs to train the model. If None, early stopping will be used.
            batch_size: batch_size

        Returns:
            self, detector trained
        """
        import tensorflow as tf

        callbacks = []
        if epochs is None:
            epochs = 100
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                min_delta=0.1,
                patience=5,
            ))

        if not self.vae:
            self.vae, self.encoder, self.decoder = self._build_vae(source.shape[-1], self.embedding_dim)

        x = source.values if isinstance(source, pd.DataFrame) else source

        hist = self.vae.fit(x, epochs=epochs, batch_size=batch_size, callbacks=callbacks)
        self.hist_ = hist

        # fit gaussian kde to embeddings
        _, _, embeddings_nodrift = self.encoder(x)
        embeddings_nodrift = embeddings_nodrift.numpy()

        # TODO: Possible future idea -> For more complicated spaces, try IFs / OCSVM
        self.kde = gaussian_kde(embeddings_nodrift.T, 'silverman')

        # Train a surroggate for replacing the kde as it can be slow for inference (+ it
        # stores train datapoints)
        self.surrogate = self._build_surrogate(inp_dim=self.kde.covariance.shape[0])
        self._fit_surrogate()

        # At this point we could actually delete self.kde
        # del(self.kde)

        return self

    def _fit_surrogate(self):
        if self.surrogate is None:
            raise RuntimeError("Model must be fitted first")

        mins = np.amin(
            self.kde.dataset,
            axis=1
        )

        maxs = np.amax(
            self.kde.dataset,
            axis=1
        )

        t = np.array(np.meshgrid(*[np.linspace(mins[i], maxs[i], num=100) for i in range(self.kde.dataset.shape[0])]))
        t = np.vstack([
            t[i, :, :].ravel() for i in range(t.shape[0] - 1, 0 - 1, -1)
        ])

        true_densities = self.kde(t)

        self.surrogate.fit(t.T, true_densities, epochs=100)

        self.max_density_point = t.T[true_densities.argmax()]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the "probability" of each sample in X being an inlier. That is, its similarity
        to the reference points seen during training.

        Args:
            X: dataset

        Returns:
            numpy array with densities
        """

        embeddings = self.predict_embeddings(X)
        densities = self.surrogate(embeddings).numpy()
        return densities

    def predict_embeddings(self, target) -> np.ndarray:
        """
        Gets the embeddings predicted by the VAE's encoder.

        Args:
            target: dataset

        Returns:
            numpy array with embeddings
        """
        x = target.values if isinstance(target, pd.DataFrame) else target
        _, _, embeddings = self.encoder(x)
        return embeddings.numpy()

    def explain(self, x: np.ndarray, ref_point: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Explains what features have to change in a certain sample (or batch of samples)
        x so it becomes a `ref_point` in the embedding space. You can use this mehod when
        you have "anomalies" and what to check what feeatures need to change for the point
        to be an inlier.

        If `ref_point` is None, the considered reference point will be the point in the
        embedding space with highest density (i.e. the most normal one).

        The returned explanation will be an array with dim(explanation) = dim(x) in which
        each item points the direction and amount each feature needs to change (also known
        as delta).

        Args:
            x: datapoint (or batch of datapoints) to explain
            ref_point: Reference point coords in the embedding space

        Returns:
            numpy array(s) with explanation(s) per item.
        """
        reference = self.max_density_point if ref_point is None else ref_point
        reconstructed_reference = self.decoder(reference[np.newaxis, ...])
        delta = reconstructed_reference - x
        return delta.numpy()

    def _build_vae(self, input_shape, latent_dim=2):
        from ._vae import VAE, Sampling  # Avoid importing TF
        import tensorflow as tf

        encoder_inputs = tf.keras.Input(shape=(input_shape))
        x = tf.keras.layers.Dense(32, activation="relu")(encoder_inputs)
        x = tf.keras.layers.Dense(16, activation="relu")(x)
        x = tf.keras.layers.Dense(8, activation="relu")(x)
        x = tf.keras.layers.Flatten()(x)
        z_mean = tf.keras.layers.Dense(latent_dim, name="z_mean")(x)
        z_log_var = tf.keras.layers.Dense(latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

        latent_inputs = tf.keras.Input(shape=(latent_dim,))
        x = tf.keras.layers.Dense(8, activation="relu")(latent_inputs)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        decoder_outputs = tf.keras.layers.Dense(input_shape, activation='linear')(x)
        decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")

        vae = VAE(encoder, decoder)
        vae.compile(optimizer=tf.keras.optimizers.Adam())

        return vae, encoder, decoder

    def _build_surrogate(self, inp_dim):
        import tensorflow as tf
        surrogate = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(inp_dim,)),
            tf.keras.layers.Dense(30, activation='tanh'),
            tf.keras.layers.Dense(20, activation='tanh'),
            tf.keras.layers.Dense(1, activation='sigmoid'),
        ])

        surrogate.compile(optimizer='adam', loss='mse')
        return surrogate