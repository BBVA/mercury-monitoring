from sklearn.utils import resample
import numpy as np


def bootstrap_eval(y_true, y_pred, eval_fn, num_resamples, resample_size=None):
    """
    Returns an array with the result of applying an evaluation function `eval_fn` on `num_resamples`
    resamples with replacement of size `resample_size` on `y_true` and `y_pred` arrays. `eval_fn` can
    be a custom function which has two arguments; `y_true` is used as a first argument when calling
    `eval_fn` and `y_pred` is used as a second argument.

    Args:
        y_true (np.array): array which will be the first argument when calling `eval_fn`. It can contain
            the labels if the `eval_fn` is defined to accept labels as the first argument.
        y_pred (np.array): array which will be the second argument when calling `eval_fn`. It can contain
            model predictions if the `eval_fn` is defined to accept predictions as the second argument.
        eval_fn (Callable): function that must return a float. Receives resamples of `y_true` as first parameter
            and resamples of `y_pred` as second parameter. For example, it can be a function that calculates
            the accuracy given `y_true` and `y_pred`, or it can be a weighted average of different metrics.
        num_resamples (int): number of resamples
        resample_size (int): size of each resample. If None, the number of samples of `y_true` and `y_pred`.

    Return:
            array of the obtained metrics for each resample


    Example:
        >>> y_true = np.array([0,0,0,0,0,1,1,1,1,1])
        >>> y_pred = np.array([0,0,0,1,1,0,0,1,1,1])
        >>> def eval_acc(y_true, y_pred):
        >>>     return np.sum(y_true == y_pred) / y_true.shape[0]
        >>> output = bootstrap_eval(y_true, y_pred, eval_fn=eval_acc, num_resamples=2000, resample_size=5)
        >>> np.mean(output)
        >>> 0.5957

    """

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("y_true and y_pred arguments must have the same number of samples")

    if resample_size is None:
        resample_size = y_true.shape[0]

    output = np.zeros((num_resamples))
    for i in range(num_resamples):
        y_true_, y_pred_ = resample(y_true, y_pred, n_samples=resample_size)
        output[i] = eval_fn(y_true_, y_pred_)

    return output