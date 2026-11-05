import numpy as np
import pytest
from scipy.constants import c

from rf_tools import wake_fit
from rf_tools.beam import GaussianBunch
from rf_tools.utils import AccuracyWarning
from rf_tools.wake import LongitudinalResonator, convolve_wake
from rf_tools.wake_fit import _find_spectral_peaks, _get_spectrum, fit_resonators

BUNCH_SIGMA = 6e-3 / c # s
WAKE_LENGTH = 10 / c # s
TIME_STEP = 2e-12 # s

# trapped modes of the pillbox in Yao et al. (table 1, eigenmode solver): frequency, R/Q, Q
NARROWBAND = [(5.843e9, 66.49, 7988), (13.558e9, 15.42, 12640), (16.108e9, 1.96, 7942)]
BROADBAND = (8e9, 1000.0, 3.0)


def make_wake(modes, wake_length=WAKE_LENGTH, bunch_sigma=BUNCH_SIGMA, noise=0.0, seed=0, start=None):
    start = -6 * bunch_sigma if start is None else start
    time = np.arange(start, wake_length, TIME_STEP)
    wake = np.zeros_like(time)
    for freq, r_over_q, q_factor in modes:
        wake += LongitudinalResonator(freq, r_over_q * q_factor, q_factor).get_wake_potential(time, bunch_sigma)
    if noise:
        tail_rms = np.sqrt(np.mean(wake[time > wake_length / 2]**2))
        wake += noise * tail_rms * np.random.default_rng(seed).standard_normal(len(time))
    return time, wake


def assert_modes_recovered(result, modes, rel=1e-6):
    assert len(result.resonators) == len(modes)
    for fit, (freq, r_over_q, q_factor) in zip(result.resonators, sorted(modes)):
        assert fit.resonant_freq == pytest.approx(freq, rel=rel)
        assert fit.r_over_q == pytest.approx(r_over_q, rel=rel)
        assert fit.q_factor == pytest.approx(q_factor, rel=rel)


# --- peak search ---

def test_window_suppresses_side_lobes_of_truncated_mode():
    time, wake = make_wake([(5e9, 50.0, 1e6)])
    bunch = GaussianBunch(charge=1, sigma=BUNCH_SIGMA)
    freq_spectrum, magnitude = _get_spectrum(time, wake, bunch, 8, 3.5, True)
    freq_peaks, _, _ = _find_spectral_peaks(freq_spectrum, magnitude, 1e-3 * np.max(magnitude), bunch)
    np.testing.assert_allclose(freq_peaks, [5e9], atol=0.01 / WAKE_LENGTH)


def test_noise_has_no_spectral_peaks():
    time = np.arange(0, WAKE_LENGTH, TIME_STEP)
    wake = np.random.default_rng(0).standard_normal(len(time))
    bunch = GaussianBunch(charge=1, sigma=BUNCH_SIGMA)
    freq, magnitude = _get_spectrum(time, wake, bunch, 8, 3.5, True)
    assert len(_find_spectral_peaks(freq, magnitude, 0, bunch)[0]) == 0


# --- the models from the specification ---

@pytest.mark.parametrize('wake_length_m', [5, 10, 60])
def test_narrowband_modes_from_partially_decayed_wake(wake_length_m):
    # none of the modes decay by more than a few percent
    result = fit_resonators(*make_wake(NARROWBAND, wake_length_m / c), BUNCH_SIGMA)
    assert_modes_recovered(result, NARROWBAND)
    assert result.residual < 1e-6


@pytest.mark.parametrize('broadband', [
    BROADBAND,
    (5.9e9, 1000.0, 3.0), # overlaps with a narrowband mode
    (2e9, 1000.0, 3.0),
    (14e9, 300.0, 3.0),
    (8e9, 30000.0, 3.0), # dominates the wake
    (8e9, 3.0, 3.0), # barely visible
    (8e9, 1000.0, 0.7), # almost overdamped
    (8e9, 100.0, 30.0),
    (0.5e9, 300.0, 3.0), # long tail
])
def test_narrowband_and_broadband_modes(broadband):
    modes = [*NARROWBAND, broadband]
    assert_modes_recovered(fit_resonators(*make_wake(modes), BUNCH_SIGMA), modes)


@pytest.mark.parametrize('use_pso_for_narrowband', [False, True])
def test_narrowband_pso_gives_the_same_result(use_pso_for_narrowband):
    modes = [*NARROWBAND, BROADBAND]
    result = fit_resonators(*make_wake(modes), BUNCH_SIGMA, use_pso_for_narrowband=use_pso_for_narrowband)
    assert_modes_recovered(result, modes)


@pytest.mark.parametrize('use_pso_for_narrowband, num_swarms', [(False, 0), (True, 1)])
def test_narrowband_pso_runs_on_request_only(monkeypatch, use_pso_for_narrowband, num_swarms):
    swarms = []
    minimize = wake_fit._minimize_with_particle_swarm
    monkeypatch.setattr(wake_fit, '_minimize_with_particle_swarm', lambda *args: swarms.append(args) or minimize(*args))
    result = fit_resonators(
        *make_wake(NARROWBAND), BUNCH_SIGMA, max_broadband_modes=0, use_pso_for_narrowband=use_pso_for_narrowband
    )
    assert len(swarms) == num_swarms
    assert_modes_recovered(result, NARROWBAND)


def test_broadband_modes_always_use_pso(monkeypatch):
    swarms = []
    minimize = wake_fit._minimize_with_particle_swarm
    monkeypatch.setattr(wake_fit, '_minimize_with_particle_swarm', lambda *args: swarms.append(args) or minimize(*args))
    fit_resonators(*make_wake([BROADBAND]), BUNCH_SIGMA, use_pso_for_narrowband=False)
    assert swarms


def test_data_that_does_not_come_from_the_model():
    # numerical convolution of the point charge wake instead of the analytic wake potential
    modes = [*NARROWBAND, BROADBAND]
    time = np.arange(0, WAKE_LENGTH + 10 * BUNCH_SIGMA, 2.5e-13)
    point_wake = sum(LongitudinalResonator(f, r * q, q).get_wake_function(time) for f, r, q in modes)
    time_convolved, wake = convolve_wake(time, point_wake, BUNCH_SIGMA)
    mask = (time_convolved >= -5 * BUNCH_SIGMA) & (time_convolved <= WAKE_LENGTH)
    assert_modes_recovered(fit_resonators(time_convolved[mask][::8], wake[mask][::8], BUNCH_SIGMA), modes, rel=1e-4)


# --- noise and model mismatch ---

def test_noise_gives_no_spurious_modes_and_consistent_errors():
    modes = [*NARROWBAND, BROADBAND]
    result = fit_resonators(*make_wake(modes, noise=1e-3), BUNCH_SIGMA)
    assert len(result.resonators) == len(modes)
    for fit, error, (freq, r_over_q, q_factor) in zip(result.resonators, result.q_factor_errors, sorted(modes)):
        assert fit.resonant_freq == pytest.approx(freq, rel=1e-4)
        assert fit.r_over_q == pytest.approx(r_over_q, rel=1e-2)
        assert abs(np.log(fit.q_factor / q_factor)) < 5 * error


def test_pure_noise_finds_no_modes():
    time = np.arange(-6 * BUNCH_SIGMA, WAKE_LENGTH, TIME_STEP)
    wake = np.random.default_rng(1).standard_normal(len(time))
    with pytest.warns(AccuracyWarning, match='No resonant modes'):
        assert fit_resonators(time, wake, BUNCH_SIGMA).resonators == ()


def test_modes_below_threshold_do_not_disturb_strong_modes():
    rng = np.random.default_rng(3)
    background = [(rng.uniform(1e9, 20e9), rng.uniform(0.01, 0.08), 10**rng.uniform(2, 4)) for _ in range(40)]
    result = fit_resonators(*make_wake([*NARROWBAND, BROADBAND, *background]), BUNCH_SIGMA)
    assert len(result.resonators) == 4
    strongest = result.resonators[0]
    assert strongest.q_factor == pytest.approx(NARROWBAND[0][2], rel=0.02)
    assert strongest.r_over_q == pytest.approx(NARROWBAND[0][1], rel=1e-3)


def test_short_wake_warns_about_q_factor():
    with pytest.warns(AccuracyWarning, match='poorly constrained'):
        result = fit_resonators(*make_wake([(5e9, 50.0, 1e6)], 1 / c, noise=1e-2), BUNCH_SIGMA)
    # the amplitude is available regardless
    assert result.resonators[0].r_over_q == pytest.approx(50, rel=1e-2)
    assert result.q_factor_errors[0] > 0.1


# --- separation of modes ---

@pytest.mark.parametrize('separation', [1.2, 1.5, 2, 5]) # in units of 1 / wake length
def test_close_modes_are_separated(separation):
    modes = [(5e9, 50.0, 8000), (5e9 + separation / WAKE_LENGTH, 50.0, 8000)]
    assert_modes_recovered(fit_resonators(*make_wake(modes), BUNCH_SIGMA), modes)


def test_weak_mode_next_to_strong_mode():
    modes = [(5e9, 50.0, 8000), (5.07e9, 1.5, 8000)]
    assert_modes_recovered(fit_resonators(*make_wake(modes), BUNCH_SIGMA), modes)


def test_beat_slower_than_wake_is_not_mistaken_for_a_decay():
    # The envelope only falls within the wake, so that the tail starts too late to separate the modes.
    # What is fitted to it extrapolates to about ten times the wake, unless the fit starts over earlier.
    modes = [(5e9, 50.0, 8000), (5e9 + 0.75 / WAKE_LENGTH, 50.0, 8000)]
    time, wake = make_wake(modes)
    # the modes are not resolved at this separation, all to be expected is a bounded and flagged model
    with pytest.warns(AccuracyWarning):
        result = fit_resonators(time, wake, BUNCH_SIGMA, use_pso_for_narrowband=True)
    assert result.long_range_start == pytest.approx(12 * BUNCH_SIGMA)
    assert result.residual < 0.1
    assert np.max(np.abs(result.get_wake_potential(time))) < 1.2 * np.max(np.abs(wake))


@pytest.mark.parametrize('use_pso_for_narrowband', [False, True])
def test_unresolved_modes_warn(use_pso_for_narrowband):
    modes = [(5e9, 50.0, 8000), (5e9 + 0.4 / WAKE_LENGTH, 50.0, 8000)]
    with pytest.warns(AccuracyWarning) as record:
        fit_resonators(*make_wake(modes), BUNCH_SIGMA, use_pso_for_narrowband=use_pso_for_narrowband)
    assert any('unexplained' in str(warning.message) for warning in record)


@pytest.mark.parametrize('separation', [0.6, 0.75])
def test_unresolved_modes_stay_bounded(separation):
    time, wake = make_wake([(5e9, 50.0, 8000), (5e9 + separation / WAKE_LENGTH, 50.0, 8000)])
    with pytest.warns(AccuracyWarning):
        result = fit_resonators(time, wake, BUNCH_SIGMA)
    assert np.max(np.abs(result.get_wake_potential(time))) < 1.5 * np.max(np.abs(wake))


# --- quality factors between narrow- and broadband ---

@pytest.mark.parametrize('q_factor', [30, 100, 300, 1000, 3000])
def test_intermediate_q_factors(q_factor):
    modes = [(5e9, 50.0, q_factor), (7e9, 50.0, 8000)]
    assert_modes_recovered(fit_resonators(*make_wake(modes), BUNCH_SIGMA), modes)


def test_mode_decayed_before_long_range_start_does_not_take_place_of_broadband_mode():
    modes = [(7.088e9, 21.5, 792), (7.936e9, 48.4, 4394), (12.017e9, 16.8, 590), (8.45e9, 17.4, 2.7)]
    result = fit_resonators(*make_wake(modes, 40 / c, bunch_sigma=7e-3 / c), 7e-3 / c)
    assert_modes_recovered(result, modes, rel=1e-3)


def test_tail_of_broadband_mode_is_not_a_narrowband_mode():
    # a low-frequency mode that has not decayed at the long-range start. Its spectral peak used to
    # end up as an additional mode at the lower bound of Q, right next to the true broadband mode
    modes = [(0.71e9, 1067.0, 5.1), (11.84e9, 30.15, 26818), (16.04e9, 16.88, 75640)]
    result = fit_resonators(*make_wake(modes, 5 / c, bunch_sigma=5.56e-3 / c), 5.56e-3 / c)
    assert_modes_recovered(result, modes, rel=1e-3)


def test_narrowband_mode_within_spectral_peak_of_broadband_mode():
    modes = [(0.779e9, 390.5, 10754), (2.259e9, 2.94, 21134), (2.19e9, 208.6, 3.24), (5.065e9, 91.95, 904)]
    result = fit_resonators(*make_wake(modes, 20 / c, bunch_sigma=12.45e-3 / c), 12.45e-3 / c)
    assert_modes_recovered(result, modes, rel=1e-3)


def test_one_mode_is_not_split_in_two():
    # decays around the long-range start, so that it is visible both in the tail and the residual
    modes = [(5222352816, 8.4784, 4984.27), (5291822858, 74.9255, 601.12), (6538048374, 135.252, 1.93), (7484898671, 43.0193, 519.74)]
    time, wake = make_wake(modes, 40 / c, bunch_sigma=8.90e-3 / c, noise=1e-2, seed=5)
    result = fit_resonators(time, wake, 8.90e-3 / c, use_pso_for_narrowband=True)
    assert len(result.resonators) == len(modes)
    assert result.resonators[-1].r_over_q == pytest.approx(43.02, rel=0.02)


# --- broadband only ---

def test_only_broadband_mode():
    assert_modes_recovered(fit_resonators(*make_wake([BROADBAND]), BUNCH_SIGMA), [BROADBAND])


def test_number_of_broadband_modes_is_limited():
    modes = [BROADBAND, (2e9, 300.0, 2.0)]
    time, wake = make_wake(modes)
    assert len(fit_resonators(time, wake, BUNCH_SIGMA, max_broadband_modes=1).resonators) == 1
    assert_modes_recovered(fit_resonators(time, wake, BUNCH_SIGMA, max_broadband_modes=2), modes)


def test_long_range_fit_only():
    result = fit_resonators(*make_wake([*NARROWBAND, BROADBAND]), BUNCH_SIGMA, max_broadband_modes=0)
    # with the tail of the broadband mode in the fit, which the refinement cannot remove
    assert_modes_recovered(result, NARROWBAND, rel=1e-3)
    assert result.long_range_residual < 1e-3


# --- result ---

def test_result_reproduces_wake_and_impedance():
    modes = [*NARROWBAND, BROADBAND]
    time, wake = make_wake(modes)
    result = fit_resonators(time, wake, BUNCH_SIGMA)
    np.testing.assert_allclose(result.get_wake_potential(time), wake, atol=1e-6 * np.max(np.abs(wake)))

    freq = np.array([NARROWBAND[0][0], 1e9])
    expected = sum(LongitudinalResonator(f, r * q, q).get_impedance(freq) for f, r, q in modes)
    np.testing.assert_allclose(result.get_impedance(freq), expected, rtol=1e-5)
    # the peak that the truncated wake underestimates
    assert result.get_impedance(freq, wake_length=WAKE_LENGTH)[0].real < 0.2 * expected[0].real


def test_resonators_are_sorted_by_freq():
    freqs = [r.resonant_freq for r in fit_resonators(*make_wake([*NARROWBAND, BROADBAND]), BUNCH_SIGMA).resonators]
    assert freqs == sorted(freqs)


# --- input ---

def test_result_does_not_depend_on_unit_of_wake():
    time, wake = make_wake([*NARROWBAND, BROADBAND], noise=1e-3)
    reference = fit_resonators(time, wake, BUNCH_SIGMA)
    scaled = fit_resonators(time, wake * 1e-12, BUNCH_SIGMA) # V/pC
    for a, b in zip(reference.resonators, scaled.resonators):
        assert b.q_factor == pytest.approx(a.q_factor, rel=1e-6)
        assert b.r_over_q == pytest.approx(a.r_over_q * 1e-12, rel=1e-6)


def test_seed_does_not_matter():
    time, wake = make_wake([*NARROWBAND, BROADBAND], noise=1e-3)
    first, second = (fit_resonators(time, wake, BUNCH_SIGMA, seed=seed) for seed in (1, 2))
    for a, b in zip(first.resonators, second.resonators):
        assert b.q_factor == pytest.approx(a.q_factor, rel=1e-5)


def test_same_seed_is_reproducible():
    time, wake = make_wake([*NARROWBAND, BROADBAND], noise=1e-3)
    first, second = (fit_resonators(time, wake, BUNCH_SIGMA, seed=7) for _ in range(2))
    assert first.resonators[0].q_factor == second.resonators[0].q_factor


def test_wake_may_start_behind_bunch_centre():
    modes = [*NARROWBAND, BROADBAND]
    assert_modes_recovered(fit_resonators(*make_wake(modes, start=0.0), BUNCH_SIGMA), modes)


def test_long_range_start_can_be_provided():
    result = fit_resonators(*make_wake(NARROWBAND), BUNCH_SIGMA, long_range_start=5e-9)
    assert result.long_range_start == 5e-9
    assert_modes_recovered(result, NARROWBAND)


def test_modes_below_peak_threshold_can_be_included():
    modes = [(5e9, 50.0, 8000), (9e9, 0.1, 8000)]
    time, wake = make_wake(modes)
    assert len(fit_resonators(time, wake, BUNCH_SIGMA, min_relative_peak_height=0.01).resonators) == 1
    assert_modes_recovered(fit_resonators(time, wake, BUNCH_SIGMA, min_relative_peak_height=1e-3), modes)


def test_wake_of_opposite_sign_warns():
    # e.g. a simulation code with the opposite convention: negative shunt impedance is not physical
    time, wake = make_wake(NARROWBAND)
    with pytest.warns(AccuracyWarning, match='sign'):
        assert fit_resonators(time, -wake, BUNCH_SIGMA).resonators == ()


def test_time_axis_with_offset_warns():
    time, wake = make_wake(NARROWBAND)
    with pytest.warns(AccuracyWarning) as record:
        fit_resonators(time + 7e-12, wake, BUNCH_SIGMA)
    assert any('bunch centre' in str(warning.message) for warning in record)


def test_rejects_invalid_input():
    time, wake = make_wake(NARROWBAND, 1 / c)
    with pytest.raises(ValueError, match='zero'):
        fit_resonators(time, 0 * wake, BUNCH_SIGMA)
    with pytest.raises(ValueError, match='samples'):
        fit_resonators(time, wake[:-1], BUNCH_SIGMA)
    with pytest.raises(ValueError, match='sigma'):
        fit_resonators(time, wake, 0)
    with pytest.raises(ValueError, match='few'):
        fit_resonators(time, wake, BUNCH_SIGMA, long_range_start=time[-1])
    with pytest.raises(ValueError, match='equidistant'):
        fit_resonators(time**2, wake, BUNCH_SIGMA)
