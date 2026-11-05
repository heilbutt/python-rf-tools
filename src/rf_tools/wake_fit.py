import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from math import pi, sqrt

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.signal.windows import kaiser

from .beam import GaussianBunch
from .units import format_quantity
from .utils import AccuracyWarning, Complex1dArray, Real1dArray, _get_uniform_step
from .wake import (
    _SHORT_RANGE_SIGMAS,
    LongitudinalResonator,
    _get_gaussian_resonator_wake_potential,
    get_impedance_from_wake,
)

logger = logging.getLogger(__name__) # rf_tools.wake_fit

__all__ = [
    'ResonatorFitResult',
    'fit_resonators'
]

# Fit of resonators to a truncated wake potential, after Yao et al., arXiv:2606.06803:
#   1. Narrowband modes are fitted to the tail of the wake, where everything else has decayed. Their
#      frequencies are peaks of the zero-padded spectrum. The fit of their quality factors converges
#      from any start, as these modes are close to orthogonal, so that the particle swarm is optional.
#   2. Broadband modes are fitted to what remains of the wake, with a particle swarm in frequency and
#      quality factor, since there is no spectral peak to rely on. These use the exact wake potential.
#   3. Least squares refine all parameters. The narrowband modes only ever see the tail, as the short-
#      range part of a simulated wake need not consist of resonators at all.
# In contrast to the paper, the amplitudes of the modes are not part of the swarm. A mode is linear in
# its amplitude, so that the best amplitude for given quality factors follows from linear least squares.
# In addition, the bunch form factor is complex (eq. 10), which matters at low quality factors.

# long-range start: segments of the wake that are compared to the level of its tail
_NUM_WAKE_SEGMENTS = 64
_MAX_LONG_RANGE_RMS_RATIO = 1.2
# A decay that starts A times above the tail is at the tail level at some time, and another factor
# of A below it at twice that time. This keeps the margin independent of the unknown decay rate.
_LONG_RANGE_START_TO_TAIL_CROSSING_RATIO = 3
_MIN_LONG_RANGE_SAMPLES = 16
_MIN_LONG_RANGE_TO_WAKE_RATIO = 1e-8 # below, the tail is numerically empty

# peak search: side lobes of this window are at -74 dB, its main lobe has a FWHM of ~1.9/T
_KAISER_BETA = 10
_KAISER_FWHM = 1.9 # in units of 1 / window length
_KAISER_MEAN = 0.391 # peak of an undamped mode in the windowed spectrum, relative to no window
_MAX_NARROWBAND_PEAK_WIDTH = 3 # in units of window FWHM, wider peaks are not narrowband modes
_MIN_PEAK_TO_MEDIAN_RATIO = 6 # white noise: maximum of 1e6 Rayleigh-distributed bins is 4.5 medians
_MIN_MODE_SEPARATION = 0.25 # in units of 1 / window length, modes cannot be told apart below this
_MAX_PEAK_SEARCH_ROUNDS = 8
_MAX_PEAKS_PER_ROUND = 16 # strongest first: the time and memory of the swarm grow quickly with its dimension

# particle swarm: constriction coefficients of Clerc and Kennedy
_PSO_INERTIA = 0.7298
_PSO_ACCELERATION = 1.4962
_PSO_MIN_PARTICLES = 24
_PSO_PARTICLES_PER_DIMENSION = 8
_PSO_MAX_ITERATIONS = 200
_PSO_STALL_ITERATIONS = 20
_PSO_STALL_TOLERANCE = 1e-5 # relative improvement of the best cost
_MAX_SEARCH_SAMPLES = 5000

# modes in what remains of the wake after the long-range fit
_MIN_BROADBAND_Q_FACTOR = 0.51
_MAX_BROADBAND_Q_FACTOR = 50 # above, it is a mode that decayed before the long-range start
# The swarm would otherwise spend itself on narrow lines, with a cost that is flat almost everywhere.
# The refinement afterwards is free to exceed this, up to the bound of the narrowband modes.
_MAX_BROADBAND_SEARCH_Q_FACTOR = 1e3
_MAX_RESIDUAL_MODES = 8 # of any quality factor
_MIN_RESIDUAL_FOR_BROADBAND = 1e-6 # relative to the wake in the fit window
_LONG_RANGE_SAMPLE_FRACTION = 0.2
_MIN_RELATIVE_SEED_HEIGHT = 0.5 # spectral peaks that are used to initialize the swarm
_MAX_SEEDS = 4

# modes are kept if they reduce the residual energy by this many times what noise allows, per parameter
_MIN_IMPROVEMENT_OVER_NOISE = 10
_MIN_RELATIVE_MODE_AMPLITUDE = 1e-6 # relative to the rms of the wake
# All modes with positive shunt impedance start in phase behind the bunch, where they cannot cancel
# each other. A single mode can therefore not exceed the maximum of the wake, not by much at least.
_MAX_MODE_AMPLITUDE_TO_WAKE_RATIO = 1.2

# least-squares refinement
_MAX_REFINEMENT_SAMPLES = 50000
_MAX_ALTERNATING_REFINEMENTS = 10
_ALTERNATING_REFINEMENT_TOLERANCE = 1e-6 # relative change of all parameters
_FREQ_DIFFERENCE_STEP = 1e-3 # in units of 1 / window length
_LOG_Q_DIFFERENCE_STEP = 1e-4

# thresholds for `AccuracyWarning`
_MAX_RELATIVE_Q_ERROR = 0.1
_MAX_LONG_RANGE_RESIDUAL = 0.02
_Q_BOUND_TOLERANCE = 1.01 # factor


@dataclass(frozen=True)
class ResonatorFitResult:

    resonators: tuple[LongitudinalResonator, ...] # sorted by frequency
    q_factor_errors: tuple[float, ...] # per resonator: formal 1 sigma error, relative to Q
    bunch_sigma: float # s
    long_range_start: float # s
    residual: float # relative to the wake, all of it
    long_range_residual: float # relative to the wake after `long_range_start`

    def get_wake_potential(
        self,
        time: Real1dArray # s
    ) -> Real1dArray:

        wake = np.zeros_like(time, dtype=float)
        for resonator in self.resonators:
            wake += resonator.get_wake_potential(time, self.bunch_sigma)
        return wake # V/C

    def get_impedance(
        self,
        freq: Real1dArray, # Hz
        wake_length: float | None = None # s
    ) -> Complex1dArray:

        impedance = np.zeros_like(freq, dtype=complex)
        for resonator in self.resonators:
            impedance += resonator.get_impedance(freq, wake_length)
        return impedance # Ohm


@dataclass(frozen=True)
class _Modes:

    freq: np.ndarray # Hz
    q_factor: np.ndarray
    r_over_q: np.ndarray # Ohm
    relative_q_error: np.ndarray

    def __len__(self) -> int:
        return len(self.freq)

    def __add__(self, other: '_Modes') -> '_Modes':
        return _Modes(
            np.concatenate([self.freq, other.freq]),
            np.concatenate([self.q_factor, other.q_factor]),
            np.concatenate([self.r_over_q, other.r_over_q]),
            np.concatenate([self.relative_q_error, other.relative_q_error])
        )

    def get_mask(self, mask: np.ndarray) -> '_Modes':
        return _Modes(self.freq[mask], self.q_factor[mask], self.r_over_q[mask], self.relative_q_error[mask])


_NO_MODES = _Modes(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0))


@dataclass(frozen=True)
class _FitSettings:

    bunch: GaussianBunch
    zero_padding_factor: float
    cutoff_by_bunch_sigma: float
    q_factor_bounds: tuple[float, float] # narrowband modes
    freq_bounds: tuple[float, float] # Hz, broadband modes
    min_mode_separation: float # Hz
    max_amplitude: float # same unit as the wake, of the long-range wake potential at t = 0
    use_pso_for_narrowband: bool
    peak_threshold: float # Ohm, of the windowed spectrum
    rng: np.random.Generator

    @property
    def bunch_sigma(self) -> float:
        return self.bunch.sigma # s

    def get_r_over_q_from_amplitude(
        self,
        amplitude: float | np.ndarray, # same unit as the wake
        freq: np.ndarray # Hz
    ) -> np.ndarray:

        return amplitude / (2 * pi * freq * self.bunch.get_form_factor(freq)) # Ohm


def _get_mode_columns(
    time: Real1dArray, # s
    freq: np.ndarray, # Hz, one per mode
    q_factor: np.ndarray, # one per mode
    bunch_sigma: float # s
) -> np.ndarray:

    # wake potential per Ohm of R/Q, with shape (num_modes, num_samples)
    return _get_gaussian_resonator_wake_potential(
        time, 2 * pi * freq[:, np.newaxis], q_factor[:, np.newaxis], bunch_sigma
    )


def _evaluate_modes(
    time: Real1dArray, # s
    modes: _Modes,
    bunch_sigma: float # s
) -> Real1dArray:

    if not len(modes):
        return np.zeros_like(time)
    return modes.r_over_q @ _get_mode_columns(time, modes.freq, modes.q_factor, bunch_sigma) # V/C


def _select_samples(
    indices: np.ndarray,
    max_samples: int,
    rng: np.random.Generator
) -> np.ndarray:

    # The modes are analytic functions of time, so the samples need not satisfy Nyquist. Random
    # instead of equidistant selection, so that two modes cannot alias onto each other.
    if len(indices) <= max_samples:
        return indices
    return np.sort(rng.choice(indices, max_samples, replace=False))


def _minimize_with_particle_swarm(
    get_costs: Callable[[np.ndarray], np.ndarray], # (num_particles, num_dimensions) -> (num_particles)
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    rng: np.random.Generator,
    initial_guesses: np.ndarray | None = None # (num_guesses, num_dimensions)
) -> np.ndarray:

    num_dimensions = len(lower_bounds)
    num_particles = _PSO_MIN_PARTICLES + _PSO_PARTICLES_PER_DIMENSION * num_dimensions
    span = upper_bounds - lower_bounds

    positions = rng.uniform(lower_bounds, upper_bounds, (num_particles, num_dimensions))
    if initial_guesses is not None:
        positions[:len(initial_guesses)] = np.clip(initial_guesses, lower_bounds, upper_bounds)
    velocities = rng.uniform(-span, span, positions.shape) / 4

    best_positions = positions.copy()
    best_costs = get_costs(positions)
    swarm_best_index = np.argmin(best_costs)
    last_improved_cost = best_costs[swarm_best_index]
    stalled_iterations = 0

    for _ in range(_PSO_MAX_ITERATIONS):
        velocities = _PSO_INERTIA * velocities + _PSO_ACCELERATION * (
            rng.random(positions.shape) * (best_positions - positions)
            + rng.random(positions.shape) * (best_positions[swarm_best_index] - positions)
        )
        positions = positions + velocities

        # particles stop at the boundary and bounce back
        is_outside = (positions < lower_bounds) | (positions > upper_bounds)
        positions = np.clip(positions, lower_bounds, upper_bounds)
        velocities[is_outside] *= -0.5

        costs = get_costs(positions)
        is_improved = costs < best_costs
        best_costs[is_improved] = costs[is_improved]
        best_positions[is_improved] = positions[is_improved]
        swarm_best_index = np.argmin(best_costs)

        if best_costs[swarm_best_index] < last_improved_cost * (1 - _PSO_STALL_TOLERANCE):
            last_improved_cost = best_costs[swarm_best_index]
            stalled_iterations = 0
        else:
            stalled_iterations += 1
            if stalled_iterations >= _PSO_STALL_ITERATIONS:
                break

    return best_positions[swarm_best_index]


def _get_costs(
    columns: np.ndarray, # (num_particles, num_modes, num_samples)
    wake: np.ndarray, # (num_samples)
    max_amplitude: float = np.inf # same unit as the wake, for columns of unit amplitude
) -> np.ndarray:

    # variable projection: a mode is linear in its amplitude, which therefore needs no search
    gram = columns @ columns.transpose(0, 2, 1)
    # columns of two modes can coincide, and low quality factors decay to zero within the samples
    diagonal = np.arange(gram.shape[1])
    gram[:, diagonal, diagonal] += 1e-10 * np.max(gram[:, diagonal, diagonal], axis=-1, keepdims=True) + 1e-300
    amplitudes = np.linalg.solve(gram, (columns @ wake)[..., np.newaxis])[..., 0]

    # physical modes only: positive shunt impedance
    amplitudes = np.clip(amplitudes, 0, max_amplitude)

    residuals = wake - (amplitudes[:, np.newaxis, :] @ columns)[:, 0, :]
    return np.sum(residuals**2, axis=-1)


def _search_narrowband_q_factors(
    time: Real1dArray, # s, long-range only
    wake: Real1dArray,
    freq: np.ndarray, # Hz, oscillation frequencies from the spectrum, kept fixed
    settings: _FitSettings
) -> np.ndarray:

    log_lower, log_upper = np.log10(settings.q_factor_bounds)
    if not settings.use_pso_for_narrowband:
        return np.full(len(freq), 10**((log_lower + log_upper) / 2))

    omega_n = 2 * pi * freq
    theta = omega_n[:, np.newaxis] * time
    cosine, sine = np.cos(theta), np.sin(theta)

    def get_costs(log_q_factors: np.ndarray) -> np.ndarray:
        # long-range wake potential, the amplitude including form factor is free and left out
        alpha = omega_n / (2 * 10**log_q_factors)
        phasor = (1 + 1j * alpha / omega_n) * np.exp(-1j * alpha * omega_n * settings.bunch_sigma**2)
        columns = np.exp(-alpha[..., np.newaxis] * time) * (
            phasor.real[..., np.newaxis] * cosine - phasor.imag[..., np.newaxis] * sine
        )
        return _get_costs(columns, wake, settings.max_amplitude)

    return 10**_minimize_with_particle_swarm(
        get_costs,
        np.full(len(freq), log_lower),
        np.full(len(freq), log_upper),
        settings.rng
    )


def _search_broadband_mode(
    time: Real1dArray, # s
    wake: Real1dArray,
    initial_guesses: np.ndarray, # (num_guesses, 2): frequency, quality factor
    settings: _FitSettings
) -> tuple[float, float]:

    def get_costs(parameters: np.ndarray) -> np.ndarray:
        columns = _get_mode_columns(time, 10**parameters[:, 0], 10**parameters[:, 1], settings.bunch_sigma)
        columns = columns[:, np.newaxis, :]
        # bunch form factor vanishes at high frequency
        scale = np.maximum(np.max(np.abs(columns), axis=-1, keepdims=True), 1e-300)
        return _get_costs(columns / scale, wake)

    best = _minimize_with_particle_swarm(
        get_costs,
        np.log10([settings.freq_bounds[0], _MIN_BROADBAND_Q_FACTOR]),
        np.log10([settings.freq_bounds[1], _MAX_BROADBAND_SEARCH_Q_FACTOR]),
        settings.rng,
        np.log10(initial_guesses) if len(initial_guesses) else None
    )
    return 10**best[0], 10**best[1]


def _get_optimal_r_over_q(
    time: Real1dArray, # s
    wake: Real1dArray,
    freq: np.ndarray, # Hz
    q_factor: np.ndarray,
    bunch_sigma: float # s
) -> np.ndarray:

    columns = _get_mode_columns(time, freq, q_factor, bunch_sigma)
    scale = np.maximum(np.max(np.abs(columns), axis=-1), 1e-300)
    amplitudes = np.linalg.lstsq(columns.T / scale, wake, rcond=None)[0]
    return np.maximum(amplitudes, 0) / scale # Ohm


def _remove_negligible_modes(
    modes: _Modes,
    wake: Real1dArray,
    settings: _FitSettings
) -> _Modes:

    min_amplitude = _MIN_RELATIVE_MODE_AMPLITUDE * sqrt(float(np.mean(wake**2)))
    return modes.get_mask(modes.r_over_q > settings.get_r_over_q_from_amplitude(min_amplitude, modes.freq))


def _get_windowed_peak_height(
    modes: _Modes,
    settings: _FitSettings
) -> np.ndarray:

    # the spectrum resolves a resonance only if it decays within the window, otherwise it is truncated
    window_length = _MIN_MODE_SEPARATION / settings.min_mode_separation
    truncated_height = modes.r_over_q * pi * modes.freq * window_length * _KAISER_MEAN
    return np.minimum(modes.r_over_q * modes.q_factor, truncated_height) # Ohm


def _get_non_coinciding_modes_mask(
    modes: _Modes,
    min_separation: float # Hz
) -> np.ndarray:

    # the stronger of two modes keeps its place
    is_kept = np.ones(len(modes), dtype=bool)
    for n in np.argsort(modes.r_over_q)[::-1]:
        if is_kept[n]:
            is_close = np.abs(modes.freq - modes.freq[n]) < min_separation
            is_close[n] = False
            is_kept[is_close & (modes.r_over_q <= modes.r_over_q[n])] = False
    return is_kept


def _is_significant_improvement(
    energy_before: float,
    energy_after: float,
    num_new_modes: int,
    num_samples: int
) -> bool:

    # each free parameter removes one sample's worth of energy from pure noise
    energy_per_parameter = energy_after / max(num_samples - 3 * num_new_modes, 1)
    return energy_before - energy_after > _MIN_IMPROVEMENT_OVER_NOISE * 3 * num_new_modes * energy_per_parameter


def _refine_modes(
    time: Real1dArray, # s
    wake: Real1dArray,
    modes: _Modes,
    freq_lower: np.ndarray, # Hz
    freq_upper: np.ndarray, # Hz
    q_factor_lower: float,
    q_factor_upper: float,
    r_over_q_upper: np.ndarray, # Ohm
    bunch_sigma: float # s
) -> _Modes:

    num_modes = len(modes)
    if not num_modes:
        return modes

    wake_scale = sqrt(float(np.mean(wake**2))) or 1.0 # keeps the tolerances of the solver relative
    freq_step = _FREQ_DIFFERENCE_STEP / (time[-1] - time[0])

    def get_residuals(x: np.ndarray) -> np.ndarray:
        columns = _get_mode_columns(time, x[:num_modes], np.exp(x[num_modes:2*num_modes]), bunch_sigma)
        return (x[2*num_modes:] @ columns - wake) / wake_scale

    def get_jacobian(x: np.ndarray) -> np.ndarray:
        freq, log_q, r_over_q = x[:num_modes], x[num_modes:2*num_modes], x[2*num_modes:]
        columns = _get_mode_columns(time, freq, np.exp(log_q), bunch_sigma)
        d_freq = _get_mode_columns(time, freq + freq_step, np.exp(log_q), bunch_sigma)
        d_log_q = _get_mode_columns(time, freq, np.exp(log_q + _LOG_Q_DIFFERENCE_STEP), bunch_sigma)
        return np.concatenate([
            ((d_freq - columns) / freq_step * r_over_q[:, np.newaxis]).T,
            ((d_log_q - columns) / _LOG_Q_DIFFERENCE_STEP * r_over_q[:, np.newaxis]).T,
            columns.T
        ], axis=1) / wake_scale

    lower = np.concatenate([freq_lower, np.full(num_modes, np.log(q_factor_lower)), np.zeros(num_modes)])
    upper = np.concatenate([freq_upper, np.full(num_modes, np.log(q_factor_upper)), r_over_q_upper])
    start = np.concatenate([modes.freq, np.log(modes.q_factor), modes.r_over_q])
    # a mode at the lower bound of R/Q has no gradient to leave it again
    start[2*num_modes:] = np.maximum(start[2*num_modes:], 1e-6 * np.max(start[2*num_modes:]))
    solution = least_squares(
        get_residuals, np.clip(start, lower, upper), jac=get_jacobian, bounds=(lower, upper),
        x_scale='jac', ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=50
    )

    # formal error of log(Q), from the Gauss-Newton Hessian and the residual level
    num_dof = max(len(time) - 3 * num_modes, 1)
    jacobian = np.asarray(solution.jac)
    variances = np.abs(np.diag(np.linalg.pinv(jacobian.T @ jacobian))) * np.sum(solution.fun**2) / num_dof

    return _Modes(
        solution.x[:num_modes],
        np.exp(solution.x[num_modes:2*num_modes]),
        solution.x[2*num_modes:],
        np.sqrt(variances[num_modes:2*num_modes])
    )


def _refine_narrowband_modes(
    time: Real1dArray, # s
    wake: Real1dArray,
    modes: _Modes,
    settings: _FitSettings
) -> _Modes:

    # the spectrum has already located the modes up to about its resolution
    margin = _KAISER_FWHM * settings.min_mode_separation / _MIN_MODE_SEPARATION
    return _refine_modes(
        time, wake, modes,
        modes.freq - margin, modes.freq + margin,
        *settings.q_factor_bounds,
        settings.get_r_over_q_from_amplitude(settings.max_amplitude, modes.freq - margin),
        settings.bunch_sigma
    )


def _refine_residual_modes(
    time: Real1dArray, # s
    wake: Real1dArray,
    modes: _Modes,
    settings: _FitSettings
) -> _Modes:

    return _refine_modes(
        time, wake, modes,
        np.full(len(modes), settings.freq_bounds[0]),
        np.full(len(modes), settings.freq_bounds[1]),
        _MIN_BROADBAND_Q_FACTOR, settings.q_factor_bounds[1],
        np.full(len(modes), np.inf),
        settings.bunch_sigma
    )


def _have_converged(
    previous: _Modes,
    modes: _Modes
) -> bool:

    for a, b in (
        (previous.freq, modes.freq),
        (previous.q_factor, modes.q_factor),
        (previous.r_over_q, modes.r_over_q)
    ):
        if np.any(np.abs(a - b) > _ALTERNATING_REFINEMENT_TOLERANCE * np.abs(b)):
            return False
    return True


def _get_long_range_start(
    time: Real1dArray, # s
    wake: Real1dArray,
    bunch_sigma: float # s
) -> float:

    # the wake is long-range where its level is no longer above the level of its tail
    segments = np.array_split(wake, _NUM_WAKE_SEGMENTS)
    segment_rms = np.array([np.sqrt(np.mean(segment**2)) for segment in segments])
    tail_rms = np.max(segment_rms[_NUM_WAKE_SEGMENTS // 2:])
    is_above_tail = segment_rms > _MAX_LONG_RANGE_RMS_RATIO * tail_rms

    crossing_time = 0.0
    if is_above_tail.any():
        crossing_segment = int(np.nonzero(is_above_tail)[0].max()) + 1
        crossing_time = float(time[sum(len(segment) for segment in segments[:crossing_segment])])

    # half of the wake always remains, and nothing of the bunch passage may enter the long-range fit
    start_time = min(_LONG_RANGE_START_TO_TAIL_CROSSING_RATIO * crossing_time, (time[0] + time[-1]) / 2)
    return float(max(start_time, _SHORT_RANGE_SIGMAS * bunch_sigma))


def _get_spectrum(
    time: Real1dArray, # s
    wake: Real1dArray,
    bunch: GaussianBunch,
    zero_padding_factor: float,
    cutoff_by_bunch_sigma: float,
    is_windowed: bool
) -> tuple[Real1dArray, Real1dArray]:

    # the window removes side lobes, which would otherwise be mistaken for modes
    freq, impedance = get_impedance_from_wake(
        time, wake * kaiser(len(wake), _KAISER_BETA) if is_windowed else wake,
        oversampling_factor=zero_padding_factor,
        bunch_sigma=bunch.sigma,
        cutoff_by_bunch_sigma=cutoff_by_bunch_sigma
    )
    return freq, np.abs(impedance) # Hz, Ohm


def _find_spectral_peaks(
    freq_spectrum: Real1dArray, # Hz
    magnitude: Real1dArray, # Ohm
    threshold: float, # Ohm
    bunch: GaussianBunch
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    # noise is white before the normalization to the bunch spectrum
    form_factor = bunch.get_form_factor(freq_spectrum)
    noise_level = _MIN_PEAK_TO_MEDIAN_RATIO * np.median(magnitude * form_factor)

    peak_indices = find_peaks(magnitude, height=threshold)[0]
    peak_indices = peak_indices[magnitude[peak_indices] * form_factor[peak_indices] > noise_level]

    freq_step = freq_spectrum[1] - freq_spectrum[0]
    freq_peaks = np.zeros(len(peak_indices))
    widths = np.zeros(len(peak_indices))
    for n, peak_index in enumerate(peak_indices):
        # quadratic interpolation of the logarithm, the main lobe is close to a gaussian
        left, centre, right = np.log(np.maximum(magnitude[peak_index - 1:peak_index + 2], 1e-300))
        curvature = left - 2 * centre + right
        offset = 0.5 * (left - right) / curvature if curvature < 0 else 0
        freq_peaks[n] = freq_spectrum[peak_index] + offset * freq_step

        # full width at half maximum
        half_maximum = magnitude[peak_index] / 2
        below_left = np.nonzero(magnitude[:peak_index] < half_maximum)[0]
        below_right = np.nonzero(magnitude[peak_index:] < half_maximum)[0]
        first = below_left[-1] if len(below_left) else 0
        last = peak_index + below_right[0] if len(below_right) else len(magnitude) - 1
        widths[n] = (last - first) * freq_step

    return freq_peaks, magnitude[peak_indices], widths # Hz, Ohm, Hz


def _fit_narrowband_modes(
    time: Real1dArray, # s, long-range only
    wake: Real1dArray,
    settings: _FitSettings
) -> _Modes:

    # wider peaks decay too fast for the tail, and are left to the fit of the residual
    max_peak_width = _MAX_NARROWBAND_PEAK_WIDTH * _KAISER_FWHM / (time[-1] - time[0])
    search_samples = _select_samples(np.arange(len(time)), _MAX_SEARCH_SAMPLES, settings.rng)
    refinement_samples = _select_samples(np.arange(len(time)), _MAX_REFINEMENT_SAMPLES, settings.rng)
    bunch_sigma = settings.bunch_sigma

    modes = _NO_MODES
    for _ in range(_MAX_PEAK_SEARCH_ROUNDS):

        # strong modes are removed from the wake, which reveals the ones that they conceal
        residual = wake - _evaluate_modes(time, modes, bunch_sigma)
        freq_peaks, heights, widths = _find_spectral_peaks(
            *_get_spectrum(time, residual, settings.bunch, settings.zero_padding_factor, settings.cutoff_by_bunch_sigma, True),
            settings.peak_threshold, settings.bunch
        )
        is_new = np.array([
            np.all(np.abs(modes.freq - freq_peak) > settings.min_mode_separation)
            for freq_peak in freq_peaks
        ], dtype=bool)
        is_candidate = (widths < max_peak_width) & is_new
        freq_peaks = freq_peaks[is_candidate][np.argsort(heights[is_candidate])[::-1][:_MAX_PEAKS_PER_ROUND]]
        if not len(freq_peaks):
            break

        logger.info('Found spectral peaks at ' + ', '.join(format_quantity(f, 'Hz') for f in freq_peaks))

        q_factors = _search_narrowband_q_factors(
            time[search_samples], residual[search_samples], freq_peaks, settings
        )
        new_modes = _Modes(
            freq_peaks, q_factors,
            np.minimum(
                _get_optimal_r_over_q(
                    time[search_samples], residual[search_samples], freq_peaks, q_factors, bunch_sigma
                ),
                settings.get_r_over_q_from_amplitude(settings.max_amplitude, freq_peaks)
            ),
            np.zeros(len(freq_peaks))
        )
        refined_modes = _remove_negligible_modes(
            _refine_narrowband_modes(time[refinement_samples], wake[refinement_samples], modes + new_modes, settings),
            wake, settings
        )
        refined_modes = refined_modes.get_mask(_get_non_coinciding_modes_mask(refined_modes, settings.min_mode_separation))

        # what wants to decay faster than allowed is not narrowband, such as the tail of a broadband
        # mode, and left to the search for broadband modes
        refined_modes = refined_modes.get_mask(
            refined_modes.q_factor > settings.q_factor_bounds[0] * _Q_BOUND_TOLERANCE
        )

        energy_before = float(np.sum(residual[refinement_samples]**2))
        energy_after = float(np.sum((wake - _evaluate_modes(time, refined_modes, bunch_sigma))[refinement_samples]**2))
        if not _is_significant_improvement(energy_before, energy_after, len(freq_peaks), len(refinement_samples)):
            break
        modes = refined_modes

    return modes


def _fit_residual_modes(
    time: Real1dArray, # s
    wake: Real1dArray, # narrowband modes removed
    samples: np.ndarray,
    total_wake_energy: float, # sum of squares of `samples`, including narrowband modes
    max_broadband_modes: int,
    settings: _FitSettings
) -> _Modes:

    # Modes of any quality factor, since this also finds those that decayed before the long-range start
    time_samples, wake_samples = time[samples], wake[samples]
    bunch_sigma = settings.bunch_sigma

    modes = _NO_MODES
    for _ in range(_MAX_RESIDUAL_MODES if max_broadband_modes else 0):
        residual = wake_samples - _evaluate_modes(time_samples, modes, bunch_sigma)
        residual_energy = float(np.sum(residual**2))
        if residual_energy <= _MIN_RESIDUAL_FOR_BROADBAND**2 * total_wake_energy:
            break

        # the spectrum only provides a starting point for a part of the swarm
        freq_spectrum, magnitude = _get_spectrum(
            time, wake - _evaluate_modes(time, modes, bunch_sigma),
            settings.bunch, settings.zero_padding_factor, settings.cutoff_by_bunch_sigma, False
        )
        freq_peaks, heights, widths = _find_spectral_peaks(
            freq_spectrum, magnitude, _MIN_RELATIVE_SEED_HEIGHT * float(np.max(magnitude)), settings.bunch
        )
        strongest = np.argsort(heights)[::-1][:_MAX_SEEDS]
        initial_guesses = np.column_stack([
            freq_peaks[strongest],
            np.maximum(freq_peaks[strongest] / np.maximum(widths[strongest], 1), _MIN_BROADBAND_Q_FACTOR)
        ])
        initial_guesses = initial_guesses[
            (initial_guesses[:, 0] > settings.freq_bounds[0]) & (initial_guesses[:, 0] < settings.freq_bounds[1])
        ]
        new_freq, new_q_factor = _search_broadband_mode(time_samples, residual, initial_guesses, settings)

        new_freq, new_q_factor = np.array([new_freq]), np.array([new_q_factor])
        new_mode = _Modes(
            new_freq, new_q_factor,
            _get_optimal_r_over_q(time_samples, residual, new_freq, new_q_factor, bunch_sigma),
            np.zeros(1)
        )
        new_modes = _refine_residual_modes(time_samples, wake_samples, modes + new_mode, settings)

        # the search can end up on a mode that is already there, or become broadband only afterwards
        new_modes = new_modes.get_mask(_get_non_coinciding_modes_mask(new_modes, settings.min_mode_separation))
        if len(new_modes) == len(modes):
            break
        if new_modes.q_factor[-1] < _MAX_BROADBAND_Q_FACTOR:
            if np.sum(modes.q_factor < _MAX_BROADBAND_Q_FACTOR) >= max_broadband_modes:
                break
        elif _get_windowed_peak_height(new_modes, settings)[-1] < settings.peak_threshold:
            break # the same threshold as for the modes from the spectrum

        new_energy = float(np.sum((wake_samples - _evaluate_modes(time_samples, new_modes, bunch_sigma))**2))
        if not _is_significant_improvement(residual_energy, new_energy, 1, len(samples)):
            break
        modes = new_modes

    return _remove_negligible_modes(modes, wake_samples, settings)


def _get_fit_quality_warnings(
    modes: _Modes,
    num_narrowband: int,
    q_factor_bounds: tuple[float, float],
    long_range_residual: float
) -> list[str]:

    if not len(modes):
        return ['No resonant modes found in wake. Check its sign (positive wake must be decelerating)']

    messages = []
    for n in range(len(modes)):
        freq = format_quantity(modes.freq[n], 'Hz')
        if modes.relative_q_error[n] > _MAX_RELATIVE_Q_ERROR:
            messages.append(
                f'Quality factor {modes.q_factor[n]:.0f} of mode at {freq} is poorly constrained (formal error {modes.relative_q_error[n]:.0%}), and so is its shunt impedance. R/Q is more reliable. Increase the wake length'
            )
        is_at_lower_bound = n < num_narrowband and modes.q_factor[n] < q_factor_bounds[0] * _Q_BOUND_TOLERANCE
        if is_at_lower_bound or modes.q_factor[n] > q_factor_bounds[1] / _Q_BOUND_TOLERANCE:
            messages.append(
                f'Quality factor of mode at {freq} is at its bound {modes.q_factor[n]:.0f}. Increase the wake length or adapt `q_factor_bounds`'
            )

    if long_range_residual > _MAX_LONG_RANGE_RESIDUAL:
        messages.append(
            f'Fitted modes leave {long_range_residual:.0%} of the long-range wake unexplained. Lower `min_relative_peak_height` or provide `long_range_start`. Modes closer than about 1 / wake length cannot be separated. Otherwise check that the bunch centre is at t = 0'
        )
    return messages


def fit_resonators(
    time: Real1dArray, # s, bunch centre at t = 0
    wake: Real1dArray, # V/C, positive wake is decelerating
    bunch_sigma: float, # s, gaussian bunch that excited the wake
    max_broadband_modes: int = 1, # with Q < 50, none disables the fit of anything but the tail
    long_range_start: float | None = None, # s, detected from the wake if not provided
    zero_padding_factor: float = 8,
    min_relative_peak_height: float = 1e-2, # impedance spectrum, relative to its strongest peak
    q_factor_bounds: tuple[float, float] = (10, 1e7), # narrowband modes
    cutoff_by_bunch_sigma: float = 3.5, # angular frequency * bunch sigma, above which the bunch does not excite modes
    use_pso_for_narrowband: bool = False, # broadband modes always use it, where it is indispensable
    seed: int = 0
) -> ResonatorFitResult:

    _get_uniform_step(time)
    if len(time) != len(wake):
        raise ValueError(f'Got {len(time)} time samples but {len(wake)} wake samples')
    if not np.any(wake):
        raise ValueError('Wake is zero everywhere')
    bunch = GaussianBunch(charge=1, sigma=bunch_sigma)

    def fit(start: float) -> tuple[ResonatorFitResult, list[str]]:
        return _fit_from_start(
            time, wake, bunch, start, max_broadband_modes, zero_padding_factor,
            min_relative_peak_height, q_factor_bounds, cutoff_by_bunch_sigma, use_pso_for_narrowband, seed
        )

    if long_range_start is not None:
        result, messages = fit(long_range_start)
    else:
        start = _get_long_range_start(time, wake, bunch_sigma)
        result, messages = fit(start)

        # The beat of two close modes looks like a decay if it is slower than the wake is long. The
        # tail is then too short for these modes, and a model that is worse than none gives it away.
        earliest_start = _SHORT_RANGE_SIGMAS * bunch_sigma
        if result.residual >= 1 and start > earliest_start:
            logger.info(f'Discarded fit with long-range start at {format_quantity(start, "s")}, retry at {format_quantity(earliest_start, "s")}')
            retried_result, retried_messages = fit(earliest_start)
            if retried_result.residual < result.residual:
                result, messages = retried_result, retried_messages

    for message in messages:
        warnings.warn(message, AccuracyWarning, stacklevel=2)
    return result


def _fit_from_start(
    time: Real1dArray, # s
    wake: Real1dArray, # V/C
    bunch: GaussianBunch,
    long_range_start: float, # s
    max_broadband_modes: int,
    zero_padding_factor: float,
    min_relative_peak_height: float,
    q_factor_bounds: tuple[float, float],
    cutoff_by_bunch_sigma: float,
    use_pso_for_narrowband: bool,
    seed: int
) -> tuple[ResonatorFitResult, list[str]]:

    bunch_sigma = bunch.sigma
    time_step = _get_uniform_step(time)
    is_long_range = time >= long_range_start
    if np.sum(is_long_range) < _MIN_LONG_RANGE_SAMPLES:
        raise ValueError(f'Long-range start at {format_quantity(long_range_start, "s")} leaves too few wake samples')

    wake_length = time[-1] - time[0]
    logger.info(f'Long-range wake starts at {format_quantity(long_range_start, "s")} of {format_quantity(wake_length, "s")}')

    time_long, wake_long = time[is_long_range], wake[is_long_range]
    max_wake = float(np.max(np.abs(wake)))
    settings = _FitSettings(
        bunch=bunch,
        zero_padding_factor=zero_padding_factor,
        cutoff_by_bunch_sigma=cutoff_by_bunch_sigma,
        q_factor_bounds=q_factor_bounds,
        freq_bounds=(1 / wake_length, min(cutoff_by_bunch_sigma / (2 * pi * bunch_sigma), 0.5 / time_step)),
        min_mode_separation=_MIN_MODE_SEPARATION / (time_long[-1] - time_long[0]),
        max_amplitude=_MAX_MODE_AMPLITUDE_TO_WAKE_RATIO * max_wake,
        use_pso_for_narrowband=use_pso_for_narrowband,
        peak_threshold=min_relative_peak_height * float(np.max(
            _get_spectrum(time_long, wake_long, bunch, zero_padding_factor, cutoff_by_bunch_sigma, True)[1]
        )),
        rng=np.random.default_rng(seed)
    )

    # narrowband modes from the tail, on which the short-range part of the wake has no influence
    has_long_range_wake = sqrt(float(np.mean(wake_long**2))) > _MIN_LONG_RANGE_TO_WAKE_RATIO * max_wake
    narrowband = _fit_narrowband_modes(time_long, wake_long, settings) if has_long_range_wake else _NO_MODES

    # all else from what remains, with all of the short-range part and some of the tail
    residual_samples = np.concatenate([
        _select_samples(np.nonzero(~is_long_range)[0], _MAX_SEARCH_SAMPLES, settings.rng),
        _select_samples(np.nonzero(is_long_range)[0], round(_LONG_RANGE_SAMPLE_FRACTION * _MAX_SEARCH_SAMPLES), settings.rng)
    ])
    residual_modes = _fit_residual_modes(
        time, wake - _evaluate_modes(time, narrowband, bunch_sigma),
        residual_samples, float(np.sum(wake[residual_samples]**2)), max_broadband_modes, settings
    )

    refinement_samples = _select_samples(np.nonzero(is_long_range)[0], _MAX_REFINEMENT_SAMPLES, settings.rng)
    for _ in range(_MAX_ALTERNATING_REFINEMENTS):
        # one mode can end up in both sets, and a pair can move together during refinement
        is_kept = _get_non_coinciding_modes_mask(narrowband + residual_modes, settings.min_mode_separation)
        narrowband, residual_modes = narrowband.get_mask(is_kept[:len(narrowband)]), residual_modes.get_mask(is_kept[len(narrowband):])
        previous_modes = narrowband + residual_modes

        narrowband = _refine_narrowband_modes(
            time[refinement_samples],
            (wake - _evaluate_modes(time, residual_modes, bunch_sigma))[refinement_samples],
            narrowband, settings
        )
        residual_modes = _refine_residual_modes(
            time[residual_samples],
            (wake - _evaluate_modes(time, narrowband, bunch_sigma))[residual_samples],
            residual_modes, settings
        )
        if _have_converged(previous_modes, narrowband + residual_modes):
            break

    narrowband = _remove_negligible_modes(narrowband, wake_long, settings)
    residual_modes = _remove_negligible_modes(residual_modes, wake[residual_samples], settings)
    modes = narrowband + residual_modes

    residual = wake - _evaluate_modes(time, modes, bunch_sigma)
    long_range_residual = float(np.linalg.norm(residual[is_long_range]) / np.linalg.norm(wake_long))
    messages = _get_fit_quality_warnings(
        modes, len(narrowband), q_factor_bounds,
        long_range_residual if has_long_range_wake else 0
    )

    order = np.argsort(modes.freq)
    return ResonatorFitResult(
        resonators=tuple(
            LongitudinalResonator(
                float(modes.freq[n]),
                float(modes.r_over_q[n] * modes.q_factor[n]),
                float(modes.q_factor[n])
            )
            for n in order
        ),
        q_factor_errors=tuple(float(modes.relative_q_error[n]) for n in order),
        bunch_sigma=bunch_sigma,
        long_range_start=long_range_start,
        residual=float(np.linalg.norm(residual) / np.linalg.norm(wake)),
        long_range_residual=long_range_residual
    ), messages
