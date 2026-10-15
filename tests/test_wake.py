from math import pi, sqrt

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf

from rf_tools.beams import Bunch
from rf_tools.utils import AccuracyWarning
from rf_tools.wake import (
    LongitudinalResonator, TransverseResonator, convolve_wake,
    get_impedance_from_wake, get_wake_from_impedance
)

F_R = 1e9 # Hz
R_SHUNT = 1e3 # Ohm
DT = 1e-12 # s
QUALITY_FACTORS = [0.6, 1, 5, 50]


def wake_time_axis(resonator, start: float = 0.0) -> np.ndarray:
    """Time axis long enough for the resonator wake to decay below ~1e-8."""
    decay_time = 1 / resonator.damping_coefficient
    return np.arange(start, 20 * decay_time, DT)


# --- resonator properties ---

@pytest.mark.parametrize('cls', [LongitudinalResonator, TransverseResonator])
def test_resonator_properties(cls):
    r = cls(F_R, R_SHUNT, 10)
    assert r.resonant_frequency == F_R
    assert r.omega_r == pytest.approx(2 * pi * F_R)
    assert r.shunt_impedance == R_SHUNT
    assert r.quality_factor == 10
    assert r.R_over_Q == pytest.approx(100)
    assert r.damping_coefficient == pytest.approx(2 * pi * F_R / 20)
    assert r.reduced_quality_factor == pytest.approx(sqrt(100 - 0.25))


@pytest.mark.parametrize('cls', [LongitudinalResonator, TransverseResonator])
@pytest.mark.parametrize('quality_factor', [0.5, 0.1, -1])
def test_resonator_rejects_overdamped(cls, quality_factor):
    with pytest.raises(ValueError):
        cls(F_R, R_SHUNT, quality_factor)


@pytest.mark.parametrize('quality_factor', QUALITY_FACTORS)
def test_resonator_reduced_omega(quality_factor):
    # oscillation frequency of the damped oscillator
    r = LongitudinalResonator(F_R, R_SHUNT, quality_factor)
    assert r.reduced_omega_r == pytest.approx(sqrt(r.omega_r**2 - r.damping_coefficient**2))


# --- longitudinal resonator ---

def test_longitudinal_impedance_special_points():
    r = LongitudinalResonator(F_R, R_SHUNT, 10)
    impedance = r.get_impedance(np.array([0, F_R / 2, F_R, 2 * F_R]))
    assert impedance[0] == 0
    assert impedance[2] == pytest.approx(R_SHUNT)
    # inductive below, capacitive above resonance
    assert impedance[1].imag > 0
    assert impedance[3].imag < 0
    assert impedance[1].real == pytest.approx(impedance[3].real)


def test_longitudinal_impedance_rejects_negative_frequencies():
    with pytest.raises(NotImplementedError):
        LongitudinalResonator(F_R, R_SHUNT, 10).get_impedance(np.array([-1.0, 0.0]))


def test_longitudinal_impedance_finite_wake_length_converges():
    r = LongitudinalResonator(F_R, R_SHUNT, 1000)
    f = np.linspace(0.99 * F_R, 1.01 * F_R, 201)
    full = r.get_impedance(f)
    long_wake = r.get_impedance(f, wake_length=100 / r.damping_coefficient)
    np.testing.assert_allclose(long_wake, full, atol=1e-3 * R_SHUNT)

    # a short wake has not yet built up the full resonance
    short_wake = r.get_impedance(np.array([F_R]), wake_length=0.1 / r.damping_coefficient)
    assert short_wake[0].real < 0.2 * R_SHUNT


@pytest.mark.parametrize('cls', [LongitudinalResonator, TransverseResonator])
def test_wake_function_is_zero_at_negative_tolerance(cls):
    r = cls(F_R, R_SHUNT, 10)
    wake = r.get_wake_function(np.array([-1e-15, -1e-16]), zero_time_tolerance=1e-15)
    assert wake[0] == 0
    if cls is LongitudinalResonator:
        assert wake[1] == pytest.approx(r.omega_r * r.R_over_Q / 2, rel=1e-6)


def test_longitudinal_wake_function_causal_and_halved_at_zero():
    r = LongitudinalResonator(F_R, R_SHUNT, 10)
    t = np.array([-1e-9, 0.0, 2e-12])
    wake = r.get_wake_function(t)
    assert wake[0] == 0
    assert wake[1] == pytest.approx(r.omega_r * r.R_over_Q / 2)
    assert wake[2] == pytest.approx(r.omega_r * r.R_over_Q, rel=2e-3)


@pytest.mark.parametrize('quality_factor', QUALITY_FACTORS)
def test_longitudinal_wake_and_impedance_are_fourier_pair(quality_factor):
    r = LongitudinalResonator(F_R, R_SHUNT, quality_factor)
    t = wake_time_axis(r)
    wake = r.get_wake_function(t)
    f, impedance = get_impedance_from_wake(t, wake)
    expected = r.get_impedance(f)
    band = (f > 0.1 * F_R) & (f < 5 * F_R)
    np.testing.assert_allclose(impedance[band], expected[band], atol=1e-4 * R_SHUNT)


def test_longitudinal_loss_factor_point_charge():
    r = LongitudinalResonator(F_R, R_SHUNT, 10)
    assert r.get_loss_factor() == pytest.approx(r.omega_r * r.R_over_Q / 2)


def test_longitudinal_loss_factor_warns_for_long_bunches():
    r = LongitudinalResonator(F_R, R_SHUNT, 10)
    with pytest.warns(AccuracyWarning, match='short bunches'):
        r.get_loss_factor(bunch_length_sigma=1 / r.omega_r)


def test_gaussian_loss_factor_time_and_frequency_domain_agree():
    # k = 1/pi * int_0^inf Re Z(w) exp(-(w sigma)^2) dw for a Gaussian bunch
    r = LongitudinalResonator(F_R, R_SHUNT, 50)
    sigma = 20e-12 # s
    k_frequency = quad(
        lambda omega: np.real(r.get_impedance(np.array([omega / 2 / pi]))[0])
        * np.exp(-(omega * sigma)**2),
        0, 2 * pi * 20 * F_R, points=[r.omega_r], limit=500
    )[0] / pi

    t = wake_time_axis(r)
    wake = r.get_wake_function(t)
    t_potential, wake_potential = convolve_wake(t, wake, target_bunch_sigma=sigma)
    k_time = Bunch(1, sigma, 'gaussian').get_loss_factor(t_potential, wake_potential)

    assert k_time == pytest.approx(k_frequency, rel=1e-6)
    # short-bunch approximation is still close, slightly beyond its validity (omega_r * sigma = 0.13)
    with pytest.warns(AccuracyWarning, match='short bunches'):
        assert r.get_loss_factor(sigma) == pytest.approx(k_frequency, rel=0.05)


# --- transverse resonator ---

def test_transverse_impedance_special_points():
    r = TransverseResonator(F_R, R_SHUNT, 10)
    impedance = r.get_impedance(np.array([0, F_R]))
    assert impedance[0] == pytest.approx(1j * r.R_over_Q)
    assert impedance[1] == pytest.approx(R_SHUNT)


def test_transverse_impedance_rejects_invalid_input():
    r = TransverseResonator(F_R, R_SHUNT, 10)
    with pytest.raises(NotImplementedError):
        r.get_impedance(np.array([-1.0, 0.0]))
    with pytest.raises(NotImplementedError):
        r.get_impedance(np.array([0.0, 1.0]), wake_length=1e-9)


def test_transverse_wake_function_causal_and_zero_at_origin():
    wake = TransverseResonator(F_R, R_SHUNT, 10).get_wake_function(np.array([-1e-9, 0.0]))
    np.testing.assert_array_equal(wake, [0, 0])


@pytest.mark.parametrize('quality_factor', QUALITY_FACTORS)
def test_transverse_wake_and_impedance_are_fourier_pair(quality_factor):
    # transverse convention: Z(w) = j * int W(t) exp(-j w t) dt
    r = TransverseResonator(F_R, R_SHUNT, quality_factor)
    t = wake_time_axis(r)
    wake = r.get_wake_function(t)
    f, impedance = get_impedance_from_wake(t, wake)
    expected = r.get_impedance(f)
    band = (f > 0.1 * F_R) & (f < 5 * F_R)
    np.testing.assert_allclose(1j * impedance[band], expected[band], atol=1e-4 * R_SHUNT)


# --- wake <-> impedance transforms ---

def test_impedance_from_wake_rejects_non_equidistant_time():
    t = np.array([0.0, 1.0, 3.0]) * DT
    with pytest.raises(ValueError):
        get_impedance_from_wake(t, np.ones(3))


def test_impedance_from_wake_is_independent_of_start_time():
    r = LongitudinalResonator(F_R, R_SHUNT, 50)
    t = wake_time_axis(r)
    wake = r.get_wake_function(t)
    wake_shifted = r.get_wake_function(t - 1e-9)
    _, impedance = get_impedance_from_wake(t, wake)
    _, impedance_shifted = get_impedance_from_wake(t - 1e-9, wake_shifted)
    np.testing.assert_allclose(impedance_shifted, impedance, atol=1e-6 * R_SHUNT)


def test_impedance_from_wake_oversampling_refines_frequency_axis():
    r = LongitudinalResonator(F_R, R_SHUNT, 50)
    t = wake_time_axis(r)
    wake = r.get_wake_function(t)
    f1, _ = get_impedance_from_wake(t, wake)
    f4, impedance = get_impedance_from_wake(t, wake, oversampling_factor=4)
    assert f4[1] == pytest.approx(f1[1] / 4)
    expected = r.get_impedance(f4)
    band = (f4 > 0.1 * F_R) & (f4 < 5 * F_R)
    np.testing.assert_allclose(impedance[band], expected[band], atol=1e-4 * R_SHUNT)


def test_impedance_from_wake_potential_normalized_by_bunch_spectrum():
    r = LongitudinalResonator(F_R, R_SHUNT, 50)
    sigma = 50e-12 # s
    t = wake_time_axis(r)
    wake = r.get_wake_function(t)
    t_potential, wake_potential = convolve_wake(t, wake, target_bunch_sigma=sigma)
    f, impedance = get_impedance_from_wake(t_potential, wake_potential, bunch_sigma=sigma)
    expected = r.get_impedance(f)
    band = (f > 0.1 * F_R) & (f < 5 * F_R)
    np.testing.assert_allclose(impedance[band], expected[band], atol=1e-3 * R_SHUNT)


def test_impedance_from_wake_cutoff_requires_bunch_sigma():
    t = np.arange(10) * DT
    with pytest.raises(ValueError):
        get_impedance_from_wake(t, np.ones(10), cutoff_by_bunch_sigma=3)


@pytest.mark.parametrize('cutoff', [2, 4])
def test_impedance_from_wake_cutoff_by_bunch_sigma(cutoff):
    # cutoff where omega * sigma = cutoff, i.e. the bunch spectrum has dropped to exp(-cutoff^2 / 2)
    r = LongitudinalResonator(F_R, R_SHUNT, 50)
    sigma = 50e-12 # s
    t = wake_time_axis(r)
    t_potential, wake_potential = convolve_wake(t, r.get_wake_function(t), target_bunch_sigma=sigma)
    f, impedance = get_impedance_from_wake(
        t_potential, wake_potential, bunch_sigma=sigma, cutoff_by_bunch_sigma=cutoff
    )
    cutoff_frequency = cutoff / (2 * pi * sigma)
    assert f[-1] <= cutoff_frequency < f[-1] + (f[1] - f[0])
    assert np.all(np.isfinite(impedance))


@pytest.mark.parametrize('num_samples', [1000, 1001])
@pytest.mark.parametrize('start_time', [0.0, -0.3e-9, 0.2e-9])
def test_wake_impedance_roundtrip(num_samples, start_time):
    rng = np.random.default_rng(42)
    t = start_time + DT * np.arange(num_samples)
    wake = rng.normal(size=num_samples)
    f, impedance = get_impedance_from_wake(t, wake)
    t_back, wake_back = get_wake_from_impedance(f, impedance, start_time=start_time)
    # an odd number of samples is padded with one zero
    np.testing.assert_allclose(t_back[:num_samples], t, rtol=0, atol=1e-6 * DT)
    np.testing.assert_allclose(wake_back[:num_samples], wake, atol=1e-12)
    np.testing.assert_allclose(wake_back[num_samples:], 0, atol=1e-12)


def test_impedance_wake_roundtrip():
    rng = np.random.default_rng(42)
    f = np.arange(501) * 1e6
    impedance = rng.normal(size=501) + 1j * rng.normal(size=501)
    # DC and Nyquist bins of a real signal are real
    impedance[[0, -1]] = impedance[[0, -1]].real
    t, wake = get_wake_from_impedance(f, impedance)
    f_back, impedance_back = get_impedance_from_wake(t, wake)
    np.testing.assert_allclose(f_back, f)
    np.testing.assert_allclose(impedance_back, impedance, atol=1e-12)


def test_wake_from_impedance_oversampling_refines_time_axis():
    f = np.arange(501) * 1e6
    impedance = LongitudinalResonator(100e6, R_SHUNT, 5).get_impedance(f)
    t1, _ = get_wake_from_impedance(f, impedance)
    t4, _ = get_wake_from_impedance(f, impedance, oversampling_factor=4)
    assert len(t4) == 4 * len(t1)
    assert t4[1] == pytest.approx(t1[1] / 4)


def test_wake_from_impedance_matches_resonator_wake():
    r = LongitudinalResonator(F_R, R_SHUNT, 5)
    f = np.arange(0, 200 * F_R, 1e6)
    impedance = r.get_impedance(f)
    t, wake = get_wake_from_impedance(f, impedance)
    assert t[0] == 0
    assert t[1] - t[0] == pytest.approx(1 / (2 * f[-1]))
    expected = r.get_wake_function(t)
    # away from the discontinuity at t = 0
    window = (t > 0.1e-9) & (t < 10e-9)
    np.testing.assert_allclose(wake[window], expected[window], atol=1e-2 * r.omega_r * r.R_over_Q)


@pytest.mark.parametrize('f', [
    np.array([1.0, 2.0, 3.0]),
    np.array([0.0, 1.0, 3.0]),
], ids=['not-starting-at-zero', 'non-equidistant'])
def test_wake_from_impedance_rejects_invalid_frequency_axis(f):
    with pytest.raises(ValueError):
        get_wake_from_impedance(f, np.ones(3, dtype=complex))


# --- convolve_wake ---

STEP_SIGMA = 10e-12 # s


def step_wake() -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(-50 * STEP_SIGMA, 50 * STEP_SIGMA, STEP_SIGMA / 20)
    return t, (t >= 0).astype(float)


def gaussian_step_response(t: np.ndarray, sigma: float) -> np.ndarray:
    return 0.5 * (1 + erf(t / (sqrt(2) * sigma)))


def test_convolve_wake_step_gives_error_function():
    t, wake = step_wake()
    t_conv, w_conv = convolve_wake(t, wake, target_bunch_sigma=3 * STEP_SIGMA)
    assert len(t_conv) == len(w_conv)
    np.testing.assert_allclose(np.diff(t_conv), t[1] - t[0])
    # only compare where the convolution is not affected by the finite input window
    window = np.abs(t_conv) < 20 * STEP_SIGMA
    np.testing.assert_allclose(
        w_conv[window], gaussian_step_response(t_conv[window], 3 * STEP_SIGMA), atol=5e-3
    )


def test_convolve_wake_from_finite_bunch_length():
    t, wake = step_wake()
    t1, w1 = convolve_wake(t, wake, target_bunch_sigma=3 * STEP_SIGMA)
    t2, w2 = convolve_wake(t1, w1, target_bunch_sigma=5 * STEP_SIGMA, wake_bunch_sigma=3 * STEP_SIGMA)
    window = np.abs(t2) < 20 * STEP_SIGMA
    np.testing.assert_allclose(
        w2[window], gaussian_step_response(t2[window], 5 * STEP_SIGMA), atol=5e-3
    )


def test_convolve_wake_warns_on_coarse_sampling():
    t, wake = step_wake()
    with pytest.warns(AccuracyWarning, match='too coarse'):
        convolve_wake(t[::40], wake[::40], target_bunch_sigma=2 * STEP_SIGMA)


def test_convolve_wake_rejects_invalid_input():
    t, wake = step_wake()
    with pytest.raises(ValueError):
        convolve_wake(t, wake, target_bunch_sigma=STEP_SIGMA, wake_bunch_sigma=STEP_SIGMA)
    with pytest.raises(ValueError):
        convolve_wake(np.array([0.0, 1.0, 3.0]) * DT, np.ones(3), target_bunch_sigma=STEP_SIGMA)
