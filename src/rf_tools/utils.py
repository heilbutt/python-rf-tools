import numpy as np

__all__ = [
    'AccuracyWarning',
    'Complex1dArray',
    'Real1dArray',
    'normalize_array',
]

Real1dArray: type = np.ndarray[tuple[int], np.dtype[np.floating]]
Complex1dArray: type = np.ndarray[tuple[int], np.dtype[np.complexfloating]]


class AccuracyWarning(UserWarning):
    """Results may be inaccurate, e.g. due to too coarse sampling or truncated data."""


def normalize_array(
    array: Real1dArray,
    minimum: float = 0,
    maximum: float = 1
) -> Real1dArray:

    return minimum + (
        (array - min(array))
        * (maximum - minimum)
        / (max(array) - min(array))
    )


def _get_uniform_step(
    array: Real1dArray,
    tolerance: float = 1e-3 # relative to step
) -> float:

    # mean step is more robust than the first difference for rounded (e.g. exported) data
    step = float(array[-1] - array[0]) / (len(array) - 1)
    uniform_array = array[0] + step * np.arange(len(array))
    if not np.allclose(array, uniform_array, rtol=0, atol=tolerance * abs(step)):
        raise ValueError('Axis must be equidistant')

    return step
