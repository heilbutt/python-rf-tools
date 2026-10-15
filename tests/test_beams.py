from math import pi, sqrt
from typing import Any

import numpy as np
import pytest
from scipy.constants import c

from rf_tools.beams import Beam, Bunch
from rf_tools.utils import AccuracyWarning

# small machine so that all tests stay fast:
# 1 us revolution period, 10 buckets of 100 ns, 1 ns sampling (1000 samples per turn)
CIRCUMFERENCE = c * 1e-6 # m
F_REV = 1e6 # Hz
MAX_FREQUENCY = 500e6 # Hz
SIGMA = 5e-9 # s
BUNCH_CHARGE = 1e-9 # C
FILLING_SCHEME = [True, True, False, True, False, False, False, False, False, False]

BUNCH_KINDS = [
    pytest.param('gaussian', None, id='gaussian'),
    pytest.param('binomial', 1.5, id='binomial-1.5'),
    pytest.param('binomial', 4.0, id='binomial-4'),
]


def make_beam(**kwargs: Any) -> Beam:
    parameters: dict[str, Any] = dict(
        circumference=CIRCUMFERENCE,
        max_frequency=MAX_FREQUENCY,
        filling_scheme=FILLING_SCHEME,
        bunch_charge=BUNCH_CHARGE,
        bunch_sigma=SIGMA,
        bunch_distribution='gaussian',
    )
    parameters.update(kwargs)
    return Beam(**parameters)


# --- Bunch ---

@pytest.mark.parametrize('distribution, exponent', BUNCH_KINDS)
def test_bunch_profile_charge_and_rms(distribution, exponent):
    bunch = Bunch(BUNCH_CHARGE, SIGMA, distribution, exponent)
    t = np.linspace(-10 * SIGMA, 10 * SIGMA, 2001)
    current = bunch.get_profile(t)
    assert np.trapezoid(current, t) == pytest.approx(BUNCH_CHARGE, rel=1e-6)
    rms = np.sqrt(np.trapezoid(t**2 * current, t) / BUNCH_CHARGE)
    assert rms == pytest.approx(SIGMA, rel=1e-4)


@pytest.mark.parametrize('exponent', [0.5, 1.5, 4.0])
def test_binomial_profile_has_finite_support(exponent):
    bunch = Bunch(1, SIGMA, 'binomial', exponent)
    half_length = SIGMA * np.sqrt(3 + 2 * exponent)
    t = np.linspace(-2 * half_length, 2 * half_length, 1001)
    current = bunch.get_profile(t)
    assert np.all(current[np.abs(t) > half_length] == 0)
    assert np.all(current[np.abs(t) < 0.99 * half_length] > 0)


@pytest.mark.parametrize('sigma, distribution, exponent', [
    (SIGMA, 'binomial', None),
    (SIGMA, 'gaussian', 2.0),
    (SIGMA, 'parabolic', None),
    (0.0, 'gaussian', None),
    (-SIGMA, 'gaussian', None),
])
def test_bunch_rejects_invalid_parameters(sigma, distribution, exponent):
    with pytest.raises(ValueError):
        Bunch(1, sigma, distribution, exponent) # type: ignore[arg-type]


@pytest.mark.parametrize('distribution, exponent', BUNCH_KINDS)
def test_bunch_form_factor_matches_numerical_spectrum(distribution, exponent):
    bunch = Bunch(BUNCH_CHARGE, SIGMA, distribution, exponent)
    f, spectrum = bunch.get_spectrum(max_frequency=2e9, frequency_step=1e6)
    # numerical spectrum of the binomial bunch converges slowly, due to its sharp edges
    np.testing.assert_allclose(
        BUNCH_CHARGE * bunch.get_form_factor(f), spectrum, atol=1e-4 * BUNCH_CHARGE
    )


def test_binomial_form_factor_uniform_distribution():
    # exponent 0 is a uniform distribution of full length 2 sqrt(3) sigma, with a sinc spectrum
    bunch = Bunch(1, SIGMA, 'binomial', 0.0)
    f = np.linspace(0, 1e9, 101)
    length = 2 * sqrt(3) * SIGMA
    np.testing.assert_allclose(bunch.get_form_factor(f), np.sinc(f * length), atol=1e-12)


@pytest.mark.parametrize('distribution, exponent', BUNCH_KINDS)
@pytest.mark.parametrize('fraction', [1e-2, 1e-4, 1e-9])
def test_bunch_half_width_and_truncated_charge(distribution, exponent, fraction):
    bunch = Bunch(1, SIGMA, distribution, exponent)
    half_width = bunch.get_half_width(fraction)
    if distribution == 'gaussian':
        assert bunch.get_truncated_charge_fraction(half_width) == pytest.approx(fraction, rel=1e-6)
    else:
        assert half_width <= SIGMA * sqrt(3 + 2 * exponent)
        assert bunch.get_truncated_charge_fraction(half_width) == pytest.approx(fraction, rel=1e-3)

    # cross-check with numerical integration
    t = np.linspace(-half_width, half_width, 20001)
    inside = np.trapezoid(bunch.get_profile(t), t)
    assert 1 - inside == pytest.approx(fraction, rel=1e-2, abs=1e-9)


def test_binomial_truncated_charge_is_zero_outside_support():
    bunch = Bunch(1, SIGMA, 'binomial', 1.5)
    assert bunch.get_truncated_charge_fraction(SIGMA * sqrt(6)) == 0
    assert bunch.get_truncated_charge_fraction(10 * SIGMA) == 0


@pytest.mark.parametrize('distribution, exponent', BUNCH_KINDS)
def test_bunch_spectrum_warns_on_coarse_sampling(distribution, exponent):
    bunch = Bunch(1, SIGMA, distribution, exponent)
    # 1 ns time step, i.e. 5 samples per sigma, is fine
    bunch.get_spectrum(max_frequency=500e6, frequency_step=1e6)
    # 5 ns time step, i.e. 1 sample per sigma, makes the bunch look like a triangle
    with pytest.warns(AccuracyWarning, match='too coarse'):
        bunch.get_spectrum(max_frequency=100e6, frequency_step=1e6)


def test_bunch_spectrum_warns_on_truncated_window():
    bunch = Bunch(1, SIGMA, 'gaussian')
    # 1 / frequency_step = 20 ns, i.e. +/- 2 sigma
    with pytest.warns(AccuracyWarning, match='truncates'):
        bunch.get_spectrum(max_frequency=1e9, frequency_step=50e6)


def test_bunch_loss_factor_warns_on_coarse_wake():
    t = np.linspace(-10 * SIGMA, 10 * SIGMA, 11)
    with pytest.warns(AccuracyWarning, match='too coarse'):
        Bunch(1, SIGMA, 'gaussian').get_loss_factor(t, np.ones_like(t))


@pytest.mark.parametrize('distribution, exponent', BUNCH_KINDS)
def test_bunch_spectrum_dc_equals_charge(distribution, exponent):
    f, spectrum = Bunch(BUNCH_CHARGE, SIGMA, distribution, exponent).get_spectrum(
        max_frequency=1e9, frequency_step=1e6
    )
    assert f[0] == 0
    assert f[1] - f[0] == pytest.approx(1e6)
    assert spectrum[0] == pytest.approx(BUNCH_CHARGE, rel=1e-9)


def test_gaussian_bunch_spectrum():
    # bunch centred at t = 0, so the spectrum is real and positive
    f, spectrum = Bunch(BUNCH_CHARGE, SIGMA, 'gaussian').get_spectrum(
        max_frequency=1e9, frequency_step=1e6
    )
    expected = BUNCH_CHARGE * np.exp(-0.5 * (2 * pi * f * SIGMA)**2)
    np.testing.assert_allclose(spectrum, expected, atol=1e-9 * BUNCH_CHARGE)


def test_bunch_loss_factor_constant_wake():
    bunch = Bunch(BUNCH_CHARGE, SIGMA, 'gaussian')
    t = np.linspace(-10 * SIGMA, 10 * SIGMA, 2001)
    assert bunch.get_loss_factor(t, np.full_like(t, 3.0)) == pytest.approx(3.0, rel=1e-6)


def test_bunch_loss_factor_pads_short_wake_with_zeros():
    # wake potential only given for t >= 0: half of a symmetric bunch sees it
    bunch = Bunch(BUNCH_CHARGE, SIGMA, 'gaussian')
    t = np.linspace(0, 10 * SIGMA, 1001)
    wake = np.ones_like(t)
    wake[0] = 0.5 # mid-point of the step, so that the trapezoidal rule is exact
    assert bunch.get_loss_factor(t, wake) == pytest.approx(0.5, rel=1e-6)


def test_bunch_loss_factor_rejects_non_equidistant_time():
    t = np.array([0.0, 1.0, 3.0]) * SIGMA
    with pytest.raises(ValueError):
        Bunch(1, SIGMA, 'gaussian').get_loss_factor(t, np.ones_like(t))


# --- Beam: construction and derived quantities ---

def test_beam_derived_quantities():
    beam = make_beam()
    assert beam.revolution_frequency == pytest.approx(F_REV)
    assert beam.revolution_period == pytest.approx(1 / F_REV)
    assert beam.harmonic_number == 10
    assert beam.bucket_length == pytest.approx(100e-9)
    assert beam.num_bunches == 3
    assert beam.charge == pytest.approx(3 * BUNCH_CHARGE)
    assert beam.average_current == pytest.approx(3 * BUNCH_CHARGE * F_REV)


def test_beam_copies_filling_scheme():
    scheme = list(FILLING_SCHEME)
    beam = make_beam(filling_scheme=scheme)
    scheme[-1] = True
    assert beam.num_bunches == 3


def test_beam_filling_scheme_validation_passes():
    make_beam(harmonic_number_validation=10, num_bunches_validation=3)


@pytest.mark.parametrize('kwargs', [
    dict(filling_scheme=[False] * 10),
    dict(harmonic_number_validation=11),
    dict(num_bunches_validation=4),
], ids=['empty', 'harmonic-number', 'num-bunches'])
def test_beam_filling_scheme_validation_fails(kwargs):
    with pytest.raises(ValueError):
        make_beam(**kwargs)


def test_beam_rejects_non_relativistic():
    with pytest.raises(NotImplementedError):
        make_beam(beta=0.9)


def test_beam_rejects_invalid_bunch_on_construction():
    with pytest.raises(ValueError):
        make_beam(bunch_distribution='binomial')


def test_beam_warns_on_coarse_sampling():
    with pytest.warns(AccuracyWarning, match='too coarse'):
        make_beam(max_frequency=100e6)


def test_beam_warns_on_bunch_too_long_for_bucket():
    # bucket is 100 ns long, i.e. only +/- 2.5 sigma
    with pytest.warns(AccuracyWarning, match='truncates'):
        make_beam(bunch_sigma=20e-9)


# --- Beam: profile and spectrum ---

@pytest.mark.parametrize('distribution, exponent', BUNCH_KINDS)
def test_beam_profile(distribution, exponent):
    beam = make_beam(bunch_distribution=distribution, bunch_binomial_exponent=exponent)
    t, current = beam.get_profile()

    dt = 1 / (2 * MAX_FREQUENCY)
    assert len(t) == 1000
    np.testing.assert_allclose(np.diff(t), dt)
    assert t[0] == pytest.approx(-beam.bucket_length / 2)
    assert np.trapezoid(current, t) == pytest.approx(beam.charge, rel=1e-6)

    # bunches sit at the centre of filled buckets, empty buckets carry no current
    per_bucket = current.reshape(beam.harmonic_number, -1)
    for bucket, is_filled in zip(per_bucket, FILLING_SCHEME):
        if is_filled:
            assert np.argmax(bucket) == len(bucket) // 2
        else:
            assert np.all(bucket == 0)


def test_beam_profile_and_spectrum_are_cached():
    beam = make_beam()
    assert beam.get_profile()[1] is beam.get_profile()[1]
    assert beam.get_spectrum()[1] is beam.get_spectrum()[1]


def test_beam_spectrum_lines_at_revolution_harmonics():
    beam = make_beam()
    f, spectrum = beam.get_spectrum()
    assert f[1] - f[0] == pytest.approx(F_REV)
    assert f[-1] == pytest.approx(MAX_FREQUENCY)
    assert spectrum[0] == pytest.approx(beam.charge, rel=1e-9)


@pytest.mark.parametrize('bucket', [0, 1, 7])
def test_beam_spectrum_phase_references_first_bucket_centre(bucket):
    # single bunch in bucket k is delayed by k bucket lengths
    scheme = [False] * 10
    scheme[bucket] = True
    beam = make_beam(filling_scheme=scheme)
    f, spectrum = beam.get_spectrum()
    expected = (
        BUNCH_CHARGE * np.exp(-0.5 * (2 * pi * f * SIGMA)**2)
        * np.exp(-2j * pi * f * bucket * beam.bucket_length)
    )
    np.testing.assert_allclose(spectrum, expected, atol=1e-9 * BUNCH_CHARGE)


def test_uniformly_filled_beam_spectrum_only_at_bunch_harmonics():
    beam = make_beam(filling_scheme=[True] * 10)
    f, spectrum = beam.get_spectrum()
    n = np.arange(len(f))
    on_bunch_harmonic = n % beam.harmonic_number == 0
    assert np.max(np.abs(spectrum[~on_bunch_harmonic])) < 1e-12 * beam.charge
    expected = beam.charge * np.exp(-0.5 * (2 * pi * f * SIGMA)**2)
    np.testing.assert_allclose(
        spectrum[on_bunch_harmonic], expected[on_bunch_harmonic],
        atol=1e-9 * beam.charge
    )


# --- Beam: power loss ---

def broadband_impedance(beam: Beam, resistance: float) -> tuple[np.ndarray, np.ndarray]:
    """Constant real impedance that excludes DC (starts above 0 Hz)."""
    f = np.linspace(beam.revolution_frequency / 2, MAX_FREQUENCY, 2000)
    return f, np.full(len(f), resistance, dtype=complex)


def test_power_loss_matches_parseval():
    # for a constant resistance R, the power lost by the AC part of the beam
    # current is R times the variance of the current over one turn
    beam = make_beam()
    resistance = 100.0 # Ohm
    _, current = beam.get_profile()
    expected = resistance * np.var(current)
    power = beam.get_power_loss(*broadband_impedance(beam, resistance))
    assert power == pytest.approx(expected, rel=1e-6)


def test_power_loss_matches_parseval_including_dc():
    # including DC, the power is R times the mean squared current
    beam = make_beam()
    resistance = 100.0 # Ohm
    _, current = beam.get_profile()
    f = np.linspace(0, MAX_FREQUENCY, 2000)
    power = beam.get_power_loss(f, np.full(len(f), resistance, dtype=complex))
    assert power == pytest.approx(resistance * np.mean(current**2), rel=1e-6)


def test_power_loss_spectrum_on_revolution_harmonics():
    beam = make_beam()
    f, spectrum = beam.get_power_loss_spectrum(*broadband_impedance(beam, 100.0))
    np.testing.assert_allclose(f, beam.get_spectrum()[0])
    assert spectrum[0] == 0
    assert np.all(spectrum >= 0)


@pytest.mark.parametrize('handling, sign', [('clip', 0), ('abs', 1)])
def test_power_loss_negative_impedance_handling(handling, sign):
    beam = make_beam()
    reference = beam.get_power_loss(*broadband_impedance(beam, 100.0))
    power = beam.get_power_loss(
        *broadband_impedance(beam, -100.0), negative_real_impedance_handling=handling
    )
    assert power == pytest.approx(sign * reference)


def test_power_loss_negative_impedance_raise():
    beam = make_beam()
    # positive impedance passes
    beam.get_power_loss(*broadband_impedance(beam, 100.0), negative_real_impedance_handling='raise')
    with pytest.raises(ValueError, match='negative'):
        beam.get_power_loss(
            *broadband_impedance(beam, -100.0), negative_real_impedance_handling='raise'
        )


def test_power_loss_unknown_negative_handling():
    beam = make_beam()
    with pytest.raises(ValueError):
        beam.get_power_loss(
            *broadband_impedance(beam, 100.0), negative_real_impedance_handling='nope' # type: ignore[arg-type]
        )


def test_power_loss_warns_on_coarse_impedance():
    beam = make_beam()
    f = np.linspace(0, MAX_FREQUENCY, 21)
    with pytest.warns(AccuracyWarning, match='coarser than revolution frequency'):
        beam.get_power_loss(f, np.ones(len(f), dtype=complex))


def test_power_loss_warns_on_impedance_not_covering_bunch_spectrum():
    beam = make_beam()
    # Gaussian bunch spectrum at 30 MHz is still 0.64 of DC
    f = np.linspace(0, 30e6, 1000)
    with pytest.warns(AccuracyWarning, match='neglected'):
        beam.get_power_loss(f, np.ones(len(f), dtype=complex))


def test_shifted_power_loss():
    beam = make_beam()
    f, beam_spectrum = beam.get_spectrum()
    # impedance at a single revolution harmonic, shifting it by s samples harmonic m + s
    resistance, m = 1e4, 100
    impedance = np.zeros(len(f), dtype=complex)
    impedance[m] = resistance
    shifts, powers = beam.get_shifted_power_loss(f, impedance, max_frequency_shift=3.5 * F_REV)

    steps = np.arange(-4, 5)
    np.testing.assert_allclose(shifts, steps * F_REV)
    expected = 2 * resistance * F_REV**2 * np.abs(beam_spectrum[m + steps])**2
    np.testing.assert_allclose(powers, expected, rtol=1e-9)
    assert powers[4] == pytest.approx(beam.get_power_loss(f, impedance))
