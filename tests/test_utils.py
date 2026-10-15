import numpy as np
import pytest

from rf_tools.utils import _get_uniform_step, normalize_array

Y = np.array([1.0, -2.0, 3.0, 0.5])


def test_normalize_array_default_range():
    np.testing.assert_allclose(normalize_array(Y), (Y - Y.min()) / (Y.max() - Y.min()))


def test_normalize_array_custom_range():
    result = normalize_array(Y, minimum=-1, maximum=3)
    assert result.min() == pytest.approx(-1)
    assert result.max() == pytest.approx(3)


@pytest.mark.parametrize('scale', [1e-12, 1.0, 1e9])
def test_uniform_step(scale):
    array = scale * (np.arange(1000) - 300) * 0.1
    assert _get_uniform_step(array) == pytest.approx(0.1 * scale)


def test_uniform_step_tolerates_rounded_values():
    # e.g. exported with a limited number of digits
    array = np.round(np.arange(1000) * 1e-3 + 1e-7 * np.sin(np.arange(1000)), 9)
    assert _get_uniform_step(array) == pytest.approx(1e-3, rel=1e-6)


@pytest.mark.parametrize('scale', [1e-12, 1.0, 1e9])
def test_uniform_step_rejects_non_equidistant(scale):
    with pytest.raises(ValueError):
        _get_uniform_step(scale * np.array([0.0, 1.0, 3.0]))
