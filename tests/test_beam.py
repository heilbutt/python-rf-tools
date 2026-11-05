from dataclasses import FrozenInstanceError, replace
from math import pi, sqrt
from typing import Any

import numpy as np
import pytest
from scipy.constants import c

from rf_tools.beam import Beam, BinomialBunch, Bunch, GaussianBunch
from rf_tools.utils import AccuracyWarning

# small machine so that all tests stay fast:
# 1 us revolution period, 10 buckets of 100 ns, 1 ns sampling (1000 samples per turn)
CIRCUMFERENCE = c * 1e-6 # m
F_REV = 1e6 # Hz
MAX_FREQ = 500e6 # Hz
SIGMA = 5e-9 # s
BUNCH_CHARGE = 1e-9 # C
FILLING_SCHEME = [True, True, False, True, False, False, False, False, False, False]

# bunches are immutable, so they can be shared between tests
GAUSSIAN_BUNCH = GaussianBunch(BUNCH_CHARGE, SIGMA)
BUNCHES = [
    pytest.param(GAUSSIAN_BUNCH, id='gaussian'),
    pytest.param(BinomialBunch(BUNCH_CHARGE, SIGMA, 1.5), id='binomial-1.5'),
    pytest.param(BinomialBunch(BUNCH_CHARGE, SIGMA, 4.0), id='binomial-4'),
]


def make_beam(**kwargs: Any) -> Beam:
    parameters: dict[str, Any] = dict(
        circumference=CIRCUMFERENCE,
        max_freq=MAX_FREQ,
        filling_scheme=FILLING_SCHEME,
        bunch=GAUSSIAN_BUNCH,
    )
    parameters.update(kwargs)
    return Beam(**parameters)


# --- Bunch ---

def test_bunch_is_abstract():
    with pytest.raises(TypeError):
        Bunch(BUNCH_CHARGE, SIGMA) # type: ignore[abstract]


def test_bunch_is_immutable():
    with pytest.raises(FrozenInstanceError):
        GAUSSIAN_BUNCH.sigma = 1.0 # type: ignore[misc]


def test_bunch_equality():
    assert GaussianBunch(BUNCH_CHARGE, SIGMA) == GAUSSIAN_BUNCH
    assert BinomialBunch(BUNCH_CHARGE, SIGMA, 1.5) != BinomialBunch(BUNCH_CHARGE, SIGMA, 2.0)


@pytest.mark.parametrize('make_bunch', [
    lambda: GaussianBunch(1, 0.0),
    lambda: GaussianBunch(1, -SIGMA),
    lambda: BinomialBunch(1, 0.0, 1.5),
    lambda: BinomialBunch(1, SIGMA, -0.5),
], ids=['gaussian-zero-sigma', 'gaussian-negative-sigma', 'binomial-zero-sigma', 'binomial-negative-exponent'])
def test_bunch_rejects_invalid_parameters(make_bunch):
    with pytest.raises(ValueError):
        make_bunch()


@pytest.mark.parametrize('bunch', BUNCHES)
def test_bunch_profile_charge_and_rms(bunch):
    t = np.linspace(-10 * SIGMA, 10 * SIGMA, 2001)
    current = bunch.get_profile(t)
    assert np.trapezoid(current, t) == pytest.approx(BUNCH_CHARGE, rel=1e-6)
    rms = np.sqrt(np.trapezoid(t**2 * current, t) / BUNCH_CHARGE)
    assert rms == pytest.approx(SIGMA, rel=1e-4)


@pytest.mark.parametrize('exponent', [0.0, 0.5, 1.5, 4.0])
def test_binomial_profile_has_finite_support(exponent):
    bunch = BinomialBunch(1, SIGMA, exponent)
    half_length = SIGMA * np.sqrt(3 + 2 * exponent)
    assert bunch.full_bunch_length == pytest.approx(2 * half_length)
    t = np.linspace(-2 * half_length, 2 * half_length, 1001)
    current = bunch.get_profile(t)
    assert np.all(current[np.abs(t) > half_length] == 0)
    assert np.all(current[np.abs(t) < 0.99 * half_length] > 0)


@pytest.mark.parametrize('bunch', BUNCHES)
def test_bunch_form_factor_matches_numerical_spectrum(bunch):
    f, spectrum = bunch.get_numeric_spectrum(max_freq=2e9, freq_step=1e6)
    # numerical spectrum of the binomial bunch converges slowly, due to its sharp edges
    np.testing.assert_allclose(
        BUNCH_CHARGE * bunch.get_form_factor(f), spectrum, atol=1e-4 * BUNCH_CHARGE
    )


def test_binomial_form_factor_uniform_distribution():
    # exponent 0 is a uniform distribution of full length 2 sqrt(3) sigma, with a sinc spectrum
    bunch = BinomialBunch(1, SIGMA, 0.0)
    f = np.linspace(0, 1e9, 101)
    length = 2 * sqrt(3) * SIGMA
    np.testing.assert_allclose(bunch.get_form_factor(f), np.sinc(f * length), atol=1e-12)


@pytest.mark.parametrize('bunch', BUNCHES)
@pytest.mark.parametrize('fraction', [1e-2, 1e-4, 1e-9])
def test_bunch_half_width_and_truncated_charge(bunch, fraction):
    half_width = bunch.get_half_width(fraction)
    if isinstance(bunch, GaussianBunch):
        assert bunch.get_truncated_charge_fraction(half_width) == pytest.approx(fraction, rel=1e-6)
    else:
        assert half_width <= bunch.full_bunch_length / 2
        assert bunch.get_truncated_charge_fraction(half_width) == pytest.approx(fraction, rel=1e-3)

    # cross-check with numerical integration
    t = np.linspace(-half_width, half_width, 20001)
    inside = np.trapezoid(bunch.get_profile(t), t) / BUNCH_CHARGE
    assert 1 - inside == pytest.approx(fraction, rel=1e-2, abs=1e-9)


def test_binomial_truncated_charge_is_zero_outside_support():
    bunch = BinomialBunch(1, SIGMA, 1.5)
    assert bunch.get_truncated_charge_fraction(bunch.full_bunch_length / 2) == 0
    assert bunch.get_truncated_charge_fraction(10 * SIGMA) == 0


@pytest.mark.parametrize('bunch', BUNCHES)
def test_bunch_numeric_spectrum_warns_on_coarse_sampling(bunch):
    # 1 ns time step, i.e. 5 samples per sigma, is fine
    bunch.get_numeric_spectrum(max_freq=500e6, freq_step=1e6)
    # 5 ns time step, i.e. 1 sample per sigma, makes the bunch look like a triangle
    with pytest.warns(AccuracyWarning, match='too coarse'):
        bunch.get_numeric_spectrum(max_freq=100e6, freq_step=1e6)


def test_bunch_numeric_spectrum_warns_on_truncated_window():
    # 1 / freq_step = 20 ns, i.e. +/- 2 sigma
    with pytest.warns(AccuracyWarning, match='truncates'):
        GAUSSIAN_BUNCH.get_numeric_spectrum(max_freq=1e9, freq_step=50e6)


def test_bunch_loss_factor_warns_on_coarse_wake():
    t = np.linspace(-10 * SIGMA, 10 * SIGMA, 11)
    with pytest.warns(AccuracyWarning, match='too coarse'):
        GAUSSIAN_BUNCH.get_loss_factor(t, np.ones_like(t))


@pytest.mark.parametrize('bunch', BUNCHES)
def test_bunch_numeric_spectrum_dc_equals_charge(bunch):
    f, spectrum = bunch.get_numeric_spectrum(max_freq=1e9, freq_step=1e6)
    assert f[0] == 0
    assert f[1] - f[0] == pytest.approx(1e6)
    assert spectrum[0] == pytest.approx(BUNCH_CHARGE, rel=1e-9)


def test_gaussian_bunch_numeric_spectrum():
    # bunch centred at t = 0, so the spectrum is real and positive
    f, spectrum = GAUSSIAN_BUNCH.get_numeric_spectrum(max_freq=1e9, freq_step=1e6)
    expected = BUNCH_CHARGE * np.exp(-0.5 * (2 * pi * f * SIGMA)**2)
    np.testing.assert_allclose(spectrum, expected, atol=1e-9 * BUNCH_CHARGE)


@pytest.mark.parametrize('bunch', BUNCHES)
def test_bunch_loss_factor_constant_wake(bunch):
    # loss factor is independent of the bunch charge
    t = np.linspace(-10 * SIGMA, 10 * SIGMA, 2001)
    assert bunch.get_loss_factor(t, np.full_like(t, 3.0)) == pytest.approx(3.0, rel=1e-6)


def test_bunch_loss_factor_pads_short_wake_with_zeros():
    # wake potential only given for t >= 0: half of a symmetric bunch sees it
    t = np.linspace(0, 10 * SIGMA, 1001)
    wake = np.ones_like(t)
    wake[0] = 0.5 # mid-point of the step, so that the trapezoidal rule is exact
    assert GAUSSIAN_BUNCH.get_loss_factor(t, wake) == pytest.approx(0.5, rel=1e-6)


def test_bunch_loss_factor_rejects_non_equidistant_time():
    t = np.array([0.0, 1.0, 3.0]) * SIGMA
    with pytest.raises(ValueError):
        GAUSSIAN_BUNCH.get_loss_factor(t, np.ones_like(t))


# --- Beam: construction and derived quantities ---

def test_beam_derived_quantities():
    beam = make_beam()
    assert beam.bunch is GAUSSIAN_BUNCH
    assert beam.revolution_freq == pytest.approx(F_REV)
    assert beam.revolution_period == pytest.approx(1 / F_REV)
    assert beam.harmonic_number == 10
    assert beam.bucket_length == pytest.approx(100e-9)
    assert beam.num_bunches == 3
    assert beam.charge == pytest.approx(3 * BUNCH_CHARGE)
    assert beam.average_current == pytest.approx(3 * BUNCH_CHARGE * F_REV)


def test_beam_is_immutable():
    beam = make_beam()
    with pytest.raises(FrozenInstanceError):
        beam.max_freq = 1e9 # type: ignore[misc]
    assert isinstance(beam.filling_scheme, tuple)


def test_beam_equality_and_replace():
    beam = make_beam()
    assert make_beam(filling_scheme=list(FILLING_SCHEME)) == beam
    modified = replace(beam, max_freq=1e9)
    assert modified.max_freq == 1e9
    assert modified.filling_scheme == beam.filling_scheme
    assert beam.max_freq == MAX_FREQ


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


def test_beam_profile_warns_on_coarse_sampling():
    # analytic spectrum is not affected by sampling, only the time-domain profile is
    beam = make_beam(max_freq=100e6)
    beam.get_spectrum()
    with pytest.warns(AccuracyWarning, match='too coarse'):
        beam.get_profile()


def test_beam_profile_warns_on_bunch_too_long_for_bucket():
    # bucket is 100 ns long, i.e. only +/- 2.5 sigma
    beam = make_beam(bunch=GaussianBunch(BUNCH_CHARGE, 20e-9))
    beam.get_spectrum()
    with pytest.warns(AccuracyWarning, match='truncates'):
        beam.get_profile()


# --- Beam: profile and spectrum ---

@pytest.mark.parametrize('bunch', BUNCHES)
def test_beam_profile(bunch):
    beam = make_beam(bunch=bunch)
    t, current = beam.get_profile()

    dt = 1 / (2 * MAX_FREQ)
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


def test_beam_squared_spectrum_matches_spectrum():
    beam = make_beam()
    f, spectrum = beam.get_spectrum()
    f_squared, squared_spectrum = beam._get_squared_spectrum()
    np.testing.assert_array_equal(f_squared, f)
    np.testing.assert_allclose(squared_spectrum, np.abs(spectrum)**2, rtol=1e-12, atol=1e-12 * beam.charge**2)


SPECTRUM_METHODS = ['get_spectrum', 'get_numeric_spectrum']


@pytest.mark.parametrize('method', SPECTRUM_METHODS)
def test_beam_spectrum_lines_at_revolution_harmonics(method):
    beam = make_beam()
    f, spectrum = getattr(beam, method)()
    np.testing.assert_allclose(f, F_REV * np.arange(len(f)))
    assert f[-1] == pytest.approx(MAX_FREQ)
    assert spectrum[0] == pytest.approx(beam.charge, rel=1e-9)


@pytest.mark.parametrize('max_freq, max_harmonic', [
    (500e6, 500),
    (500.7e6, 500),
    (499.3e6, 499),
])
def test_beam_spectrum_up_to_max_freq(max_freq, max_harmonic):
    f, _ = make_beam(max_freq=max_freq).get_spectrum()
    assert len(f) == max_harmonic + 1


@pytest.mark.parametrize('bunch', BUNCHES)
@pytest.mark.parametrize('filling_scheme', [
    FILLING_SCHEME,
    [True] * 10,
    [False] * 9 + [True],
], ids=['mixed', 'full', 'last-bucket'])
def test_beam_spectrum_matches_numeric_spectrum(bunch, filling_scheme):
    beam = make_beam(bunch=bunch, filling_scheme=filling_scheme)
    f, spectrum = beam.get_spectrum()
    f_numeric, spectrum_numeric = beam.get_numeric_spectrum()
    np.testing.assert_allclose(f_numeric[:len(f)], f)
    # numeric spectrum of the binomial bunch converges slowly, due to its sharp edges,
    # but is bounded by the aliasing threshold of the sampling check
    tolerance = 1e-12 if isinstance(bunch, GaussianBunch) else 1e-3
    np.testing.assert_allclose(spectrum_numeric[:len(f)], spectrum, atol=tolerance * beam.charge)


@pytest.mark.parametrize('method', SPECTRUM_METHODS)
@pytest.mark.parametrize('bucket', [0, 1, 7])
def test_beam_spectrum_phase_references_first_bucket_centre(method, bucket):
    # single bunch in bucket k is delayed by k bucket lengths
    scheme = [False] * 10
    scheme[bucket] = True
    beam = make_beam(filling_scheme=scheme)
    f, spectrum = getattr(beam, method)()
    expected = (
        BUNCH_CHARGE * np.exp(-0.5 * (2 * pi * f * SIGMA)**2)
        * np.exp(-2j * pi * f * bucket * beam.bucket_length)
    )
    np.testing.assert_allclose(spectrum, expected, atol=1e-9 * BUNCH_CHARGE)


@pytest.mark.parametrize('method', SPECTRUM_METHODS)
def test_uniformly_filled_beam_spectrum_only_at_bunch_harmonics(method):
    beam = make_beam(filling_scheme=[True] * 10)
    f, spectrum = getattr(beam, method)()
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
    f = np.linspace(beam.revolution_freq / 2, MAX_FREQ, 2000)
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
    f = np.linspace(0, MAX_FREQ, 2000)
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
    f = np.linspace(0, MAX_FREQ, 21)
    with pytest.warns(AccuracyWarning, match='coarser than revolution frequency'):
        beam.get_power_loss(f, np.ones(len(f), dtype=complex))


def test_power_loss_warns_on_impedance_not_covering_bunch_spectrum():
    beam = make_beam()
    # Gaussian bunch spectrum at 30 MHz is still 0.64 of DC
    f = np.linspace(0, 30e6, 1000)
    with pytest.warns(AccuracyWarning, match='highest impedance frequency'):
        beam.get_power_loss(f, np.ones(len(f), dtype=complex))


def test_power_loss_warns_on_beam_not_covering_bunch_spectrum():
    # Gaussian bunch spectrum at 50 MHz is still 0.29 of DC
    beam = make_beam(max_freq=50e6)
    with pytest.warns(AccuracyWarning, match='maximum frequency of the beam'):
        beam.get_power_loss(*broadband_impedance(beam, 100.0))


def test_shifted_power_loss_matches_brute_force():
    beam = make_beam()
    f, beam_spectrum = beam.get_spectrum()
    rng = np.random.default_rng(42)
    # impedance on the harmonics, non-zero at DC, shifts beyond the spectrum length
    impedance = rng.uniform(1, 100, size=len(f)).astype(complex)
    num_harmonics = len(f)
    shifts, powers = beam.get_shifted_power_loss(
        f, impedance, max_freq_shift=(num_harmonics + 2) * F_REV
    )

    squared_current = F_REV**2 * np.abs(beam_spectrum)**2
    for shift, power in zip(np.round(shifts / F_REV).astype(int), powers):
        shifted = np.zeros(num_harmonics)
        for n in range(num_harmonics):
            if 0 <= n - shift < num_harmonics:
                shifted[n] = impedance[n - shift].real
        expected = 2 * np.sum(squared_current * shifted) - squared_current[0] * shifted[0]
        # FFT correlation has round-off errors relative to the largest power
        assert power == pytest.approx(expected, rel=1e-9, abs=1e-12 * float(np.max(powers)))


def test_shifted_power_loss():
    beam = make_beam()
    f, beam_spectrum = beam.get_spectrum()
    # impedance at a single revolution harmonic, shifting it by s samples harmonic m + s
    resistance, m = 1e4, 100
    impedance = np.zeros(len(f), dtype=complex)
    impedance[m] = resistance
    shifts, powers = beam.get_shifted_power_loss(f, impedance, max_freq_shift=3.5 * F_REV)

    steps = np.arange(-4, 5)
    np.testing.assert_allclose(shifts, steps * F_REV)
    expected = 2 * resistance * F_REV**2 * np.abs(beam_spectrum[m + steps])**2
    np.testing.assert_allclose(powers, expected, rtol=1e-9)
    assert powers[4] == pytest.approx(beam.get_power_loss(f, impedance))
