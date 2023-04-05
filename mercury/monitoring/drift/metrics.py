import numpy as np

from scipy.special import rel_entr
from ._resampling import bootstrap_eval

from .ks_drift_detector import KSDrift


def bootstrap_eval_drift(
    y_true_src=None, y_pred_src=None, y_true_target=None, y_pred_target=None, eval_fn=None,
    resample_size_src=None, resample_size_target=None, num_resamples=50, drift_detector=None
):

    """
    This function performs resampling with replacement for a source set and a target set to obtain
    a distribution of some evaluation metric for both sets. Then, applies a drift detector to calculate
    the drift of the metric between the source and target sets. More concretely, first performs resampling
    with replacement in the arrays `y_true_src` and `y_pred_src` and calculates a metric using the
    `eval_fn` function in order to obtain a metric for each resample. Thus, we obtain a distribution of a
    metric for the source set. Next, it performs the same procedure with the arrays `y_true_target` and
    `y_pred_target`, which gives a distribution of the metric for the target dataset. Finally, applies
    a drift detector (KSDrift if not specified) in order to calculate the drift between the source and
    target obtained distributions.

    Args:
        y_true_src (np.array): array which will be the first argument when calling `eval_fn` for the source
            dataset. It can contain the labels of the source dataset if `eval_fn` is defined to accept
            labels as the first argument.
        y_pred_src (np.array): array which will be the second argument when calling `eval_fn` for the source
            dataset. It can contain the source dataset model predictions if the `eval_fn` is defined to
            accept predictions as the second argument.
        y_true_target (np.array): array which will the first argument when calling `eval_fn` for the target
            dataset. It can contain the labels of the target dataset if `eval_fn` is defined to accept
            labels as the first argument.
        y_pred_target (np.array): array which will be the second argument when calling `eval_fn` for the
            target dataset. It can contain the target dataset model predictions if the `eval_fn` is defined
            to accept predictions as the second argument.
        eval_fn (Callable): function that must return a float. For the source dataset, receives a resamples
            of `y_true_src` as first parameter and resamples of `y_pred` as second parameter. Similarly, it
            uses resamples of `y_true_target` as first parameter and resamples of `y_pred_target` as second
            parameter for the target dataset. For example, it can be a function that calculates the accuracy
            given `y_true` and `y_pred`, or it can be a weighted average of different metrics.
        resample_size_src (int): size of each resample for the source dataset. If None, length of the source.
        resample_size_target (int): size of each resample for the target dataset. If None, length of the target.
        num_resamples (int): number of resamples
        drift_detector: drift detector to use. If None it will create a KSDrift object

    Returns:
            drift_metrics (dict): dictionary with the obtained drift metrics
            drift_detector (BaseBatchDriftDetector): drift detector object used to calculate drift
            dist_src (np.array): obtained metrics distribution for the source dataset
            dist_target (np.array): obtained metrics distribution for the target dataset

    Example:
        ```python
        >>> y_true_src = np.array([0,0,0,0,0,1,1,1,1,1])
        >>> y_pred_src = np.array([0,0,0,0,1,0,1,1,1,1])
        >>> y_true_target = np.array([0,0,0,0,0,1,1,1,1,1])
        >>> y_pred_target = np.array([0,1,1,1,1,0,0,0,0,1])
        >>> def eval_acc(y_true, y_pred):
        ...     return np.sum(y_true == y_pred) / y_true.shape[0]

        >>> drift_metrics, drift_detector, dist_src, dist_target = bootstrap_eval_drift(
        ...        y_true_src=y_true_src, y_pred_src=y_pred_src, y_true_target=y_true_target, y_pred_target=y_pred_target,
        ...        resample_size_src=8, resample_size_target=8, num_resamples=1000,
        ...        eval_fn=eval_acc, drift_detector=None
        ...    )
        >>> drift_metrics["drift_detected]
        True
        >>> drift_metrics["score"]
        0.943
        ```
    """

    if resample_size_src is None:
        resample_size_src = len(y_true_src)
    if resample_size_target is None:
        resample_size_target = len(y_true_target)

    # Get distributions for source dataset
    dist_src = bootstrap_eval(y_true_src, y_pred_src, eval_fn, num_resamples, resample_size_src)

    # Get distributions for target dataset
    dist_target = bootstrap_eval(y_true_target, y_pred_target, eval_fn, num_resamples, resample_size_target)

    # Apply drift detector. If it is not passed, use ks drift
    if drift_detector is None:
        drift_detector = KSDrift(
            X_src=dist_src.reshape(-1, 1), X_target=dist_target.reshape(-1, 1), p_val=0.001, correction="bonferroni"
        )
    else:
        drift_detector.set_datasets(X_src=dist_src.reshape(-1, 1), X_target=dist_target.reshape(-1, 1))
    drift_metrics = drift_detector.calculate_drift()

    return drift_metrics, drift_detector, dist_src, dist_target


def hellinger_distance(p, q, normalize=True):
    """
    Computes Hellinger distance between two histograms as specified in this paper:
    http://users.rowan.edu/~polikar/RESEARCH/PUBLICATIONS/cidue11.pdf
    Both histograms are represented by numpy arrays with the same number of dimensions,
    where each dimension represents the counts for that particular bin.
    It is assumed that the bin edges are the same.

    Args:
        p (np.array): First histogram. Each dimension represents one bin and bin edges are assumed to be the same as in q.
        q (np.array): Second histogram. Each dimension represents one bin and bin edges are assumed to be the same as in p.
        normalize (bool): Whether to normalize the histograms. If True the proportions of each bin are calculated first
            and then the distance is calculated with the proportions

        Returns:
            (float): float representing the Hellinger distance

    """

    if len(p) != len(q):
        raise ValueError("p and q must have the same size and represent the same bins")

    if normalize:
        p = p / np.sum(p)
        q = q / np.sum(q)

    distances = (np.sqrt(p) - np.sqrt(q)) ** 2
    return np.sqrt(np.sum(distances))


def jeffreys_divergence(p, q, normalize=True):

    """
    Computes Jeffreys divergence between two histograms as specified in this paper:
    https://www.cs.cmu.edu/~efros/courses/LBMV07/Papers/rubner-jcviu-00.pdf
    Both histograms are represented by numpy arrays with the same number of dimensions,
    where each dimension represents the counts for that particular bin.
    It is assumed that the bin edges are the same.

    Args:
        p (np.array): First histogram. Each dimension represents one bin and bin edges are assumed to be the same as in q.
        q (np.array): Second histogram. Each dimension represents one bin and bin edges are assumed to be the same as in p.
        normalize (bool): Whether to normalize the histograms. If True the proportions of each bin are calculated first
            and then the distance is calculated with the proportions

        Returns:
            (float): float representing the hellinger distance

    """

    if len(p) != len(q):
        raise ValueError("p and q must have the same size and represent the same bins")

    if normalize:
        p = p / np.sum(p)
        q = q / np.sum(q)

    m = (p + q) / 2
    distances = rel_entr(p, m) + rel_entr(q, m)
    return np.sum(distances)


def psi(p, q, normalize=True, eps=1e-4):

    """
    Calculates the Population Stability Index (PSI). The metric helps to measure the stability between two population
    samples. It assumes that the two population samples have already been splitted in bins, so the histograms are the
    input to this function.

    Args:
        p (np.array): First histogram. Each dimension represents one bin and bin edges are assumed to be the same as in q.
        q (np.array): Second histogram. Each dimension represents one bin and bin edges are assumed to be the same as in p.
        normalize (bool): Whether to normalize the histograms. If True the proportions of each bin are calculated first
            and then the distance is calculated with the proportions

    Returns:
        (float): float representing the PSI

    Example:
        ```python
        >>> a = np.array([12, 11, 14, 12, 12, 10, 12, 6, 6, 5])
        >>> b = np.array([11, 11, 12, 13, 11, 11, 13, 5, 7, 6])
        >>> psi = psi(a, b)
        ```
    """

    if len(p.shape) != 1 or len(q.shape) != 1:
        raise ValueError("p and q must be np.array with len(shape)==1")

    if len(p) != len(q):
        raise ValueError("p and q must have the same size and represent the same bins")

    if normalize is None:
        if np.any(p > 1) or np.any(q > 1):
            normalize = True
        else:
            normalize = False

    if normalize:
        p = p / np.sum(p)
        q = q / np.sum(q)

    # Replace 0's to avoid inf and nans
    p[p == 0] = eps
    q[q == 0] = eps

    psi = (p - q) * np.log(p / q)
    return np.sum(psi)