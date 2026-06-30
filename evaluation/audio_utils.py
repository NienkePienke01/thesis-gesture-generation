import librosa
from librosa import ParameterError
import numpy as np
from torch import Callable, Optional

def valid_int(x: float, *, cast: Optional[Callable[[float], float]] = None) -> int:
    """Ensure that an input value is integer-typed.
    This is primarily useful for ensuring integrable-valued
    array indices.

    Parameters
    ----------
    x : number
        A scalar value to be cast to int
    cast : function [optional]
        A function to modify ``x`` before casting.
        Default: `np.floor`

    Returns
    -------
    x_int : int
        ``x_int = int(cast(x))``

    Raises
    ------
    ParameterError
        If ``cast`` is provided and is not callable.
    """
    if cast is None:
        cast = np.floor

    if not callable(cast):
        raise ParameterError("cast parameter must be callable")

    return int(cast(x))

def __peak_pick(x, pre_max, post_max, pre_avg, post_avg, delta, wait, peaks):
    """Vectorized wrapper for the peak-picker"""
    # Special case the first frame
    peaks[0] = (x[0] >= np.max(x[:min(post_max, x.shape[0])]))
    peaks[0] &= (x[0] >= np.mean(x[:min(post_avg, x.shape[0])]) + delta)

    if peaks[0]:
        n = wait + 1
    else:
        n = 1

    while n < x.shape[0]:
        maxn = np.max( x[max(0, n-pre_max):min(n+post_max, x.shape[0])])

        # Are we the local max and sufficiently above average?
        peaks[n] = (x[n] == maxn) 
        
        if not peaks[n]:
            n += 1
            continue

        avgn = np.mean(x[max(0, n-pre_avg):min(n+post_avg, x.shape[0])])
        peaks[n] &= (x[n] >= avgn + delta)

        if not peaks[n]:
            n += 1
            continue

        # Skip the next `wait` frames
        n += wait + 1


def peak_pick(
    x: np.ndarray,
    *,
    pre_max: int,
    post_max: int,
    pre_avg: int,
    post_avg: int,
    delta: float,
    wait: int,
    sparse: bool = True,
    axis: int = -1
) -> np.ndarray:
    """Use a flexible heuristic to pick peaks in a signal.

    A sample n is selected as an peak if the corresponding ``x[n]``
    fulfills the following three conditions:

    1. ``x[n] == max(x[n - pre_max:n + post_max])``
    2. ``x[n] >= mean(x[n - pre_avg:n + post_avg]) + delta``
    3. ``n - previous_n > wait``

    where ``previous_n`` is the last sample picked as a peak (greedily).

    This implementation is based on [#]_ and [#]_.

    .. [#] Boeck, Sebastian, Florian Krebs, and Markus Schedl.
        "Evaluating the Online Capabilities of Onset Detection Methods." ISMIR.
        2012.

    .. [#] https://github.com/CPJKU/onset_detection/blob/master/onset_program.py

    Parameters
    ----------
    x : np.ndarray
        input signal to peak picks from
    pre_max : int >= 0 [scalar]
        number of samples before ``n`` over which max is computed
    post_max : int >= 1 [scalar]
        number of samples after ``n`` over which max is computed
    pre_avg : int >= 0 [scalar]
        number of samples before ``n`` over which mean is computed
    post_avg : int >= 1 [scalar]
        number of samples after ``n`` over which mean is computed
    delta : float >= 0 [scalar]
        threshold offset for mean
    wait : int >= 0 [scalar]
        number of samples to wait after picking a peak
    sparse : bool [scalar]
        If `True`, the output are indices of detected peaks.
        If `False`, the output is a dense boolean array of the same
        shape as ``x``.
    axis : int [scalar]
        the axis over which to detect peaks.

    Returns
    -------
    peaks : np.ndarray [shape=(n_peaks,) or shape=x.shape, dtype=int or bool]
        indices of peaks in ``x`` (sparse=True)
        or a boolean array where `peaks[..., n]` indicates a peak at frame index `n` (sparse=False)

    Raises
    ------
    ParameterError
        If any input lies outside its defined range

    Examples
    --------
    >>> y, sr = librosa.load(librosa.ex('trumpet'))
    >>> onset_env = librosa.onset.onset_strength(y=y, sr=sr,
    ...                                          hop_length=512,
    ...                                          aggregate=np.median)
    >>> peaks = librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=0.5, wait=10)
    >>> peaks
    array([  3,  27,  40,  61,  72,  88, 103])

    Using dense output to make a boolean array of peak indicators
    >>> librosa.util.peak_pick(onset_env, pre_max=3, post_max=3, pre_avg=3, post_avg=5,
    ...                        delta=0.5, wait=10, sparse=False)
    array([False, False, ..., False, False])

    >>> import matplotlib.pyplot as plt
    >>> times = librosa.times_like(onset_env, sr=sr, hop_length=512)
    >>> fig, ax = plt.subplots(nrows=2, sharex=True)
    >>> D = np.abs(librosa.stft(y))
    >>> librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max),
    ...                          y_axis='log', x_axis='time', ax=ax[1])
    >>> ax[0].plot(times, onset_env, alpha=0.8, label='Onset strength')
    >>> ax[0].vlines(times[peaks], 0,
    ...              onset_env.max(), color='r', alpha=0.8,
    ...              label='Selected peaks')
    >>> ax[0].legend(frameon=True, framealpha=0.8)
    >>> ax[0].label_outer()
    """
    if pre_max < 0:
        raise ParameterError("pre_max must be non-negative")
    if pre_avg < 0:
        raise ParameterError("pre_avg must be non-negative")
    if delta < 0:
        raise ParameterError("delta must be non-negative")
    if wait < 0:
        raise ParameterError("wait must be non-negative")
    if post_max <= 0:
        raise ParameterError("post_max must be positive")
    if post_avg <= 0:
        raise ParameterError("post_avg must be positive")
    if sparse and x.ndim != 1:
        raise ParameterError(f"sparse=True (default) does not support "
                f"{x.ndim}-dimensional inputs. "
                f"Either set sparse=False or process each dimension independently.")

    # Ensure valid index types
    pre_max = valid_int(pre_max, cast=np.ceil)
    post_max = valid_int(post_max, cast=np.ceil)
    pre_avg = valid_int(pre_avg, cast=np.ceil)
    post_avg = valid_int(post_avg, cast=np.ceil)
    wait = valid_int(wait, cast=np.ceil)

    peaks = np.zeros_like(x, dtype=bool)
    __peak_pick(x.swapaxes(axis, -1), pre_max, post_max, pre_avg, post_avg, delta, wait, peaks.swapaxes(axis, -1))

    if sparse:
        return np.flatnonzero(peaks)

    return peaks