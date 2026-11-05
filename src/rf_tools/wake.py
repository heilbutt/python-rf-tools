import logging
import warnings
from math import ceil, pi, sqrt

import numpy as np
from scipy.constants import c
from scipy.signal import fftconvolve
from scipy.special import wofz

from .beam import GaussianBunch
from .utils import AccuracyWarning, Complex1dArray, Real1dArray, _get_uniform_step

logger = logging.getLogger(__name__) # rf_tools.wake

__all__ = [
    'LongitudinalResonator',
    'TransverseResonator',
    'convolve_wake',
    'get_impedance_from_wake',
    'get_wake_from_impedance'
]

# threshold for `AccuracyWarning`: validity of the short-bunch loss factor approximation
_MAX_OMEGA_SIGMA_SHORT_BUNCH = 0.1
# beyond this distance from the bunch centre, the short-range wake potential term is below 1e-30
_SHORT_RANGE_SIGMAS = 12


def get_impedance_from_wake(
    time: Real1dArray, # s
    wake: Real1dArray, # V/C/m^n, n=(2p) for longitudinal, n=(2p-1) for transverse
    oversampling_factor: float = 1,
    bunch_sigma: float | None = None, # s
    cutoff_by_bunch_sigma: float | None = None,
) -> tuple[Real1dArray, Complex1dArray]:

    dt = _get_uniform_step(time)

    # even number of samples, so that `get_wake_from_impedance` is the exact inverse
    num_samples = 2 * ceil(len(time) * oversampling_factor / 2)
    freq = np.fft.rfftfreq(num_samples, d=dt) # Hz

    # convention: positive wake is decelerating <-> positive impedance
    impedance = np.fft.rfft(wake, n=num_samples) * dt # Ohm/m^n

    # correct for non-zero start time
    impedance *= np.exp(-2j * pi * freq * time[0])

    # cutoff before normalization, avoids division by the vanishing bunch spectrum
    if cutoff_by_bunch_sigma is not None:
        if bunch_sigma is None:
            raise ValueError('Cannot cutoff by bunch sigma if `bunch_sigma` is not provided')

        # in angular frequency, the gaussian bunch spectrum has rms width 1/sigma
        cutoff_freq = cutoff_by_bunch_sigma / (2 * pi * bunch_sigma) # Hz
        mask = freq <= cutoff_freq
        freq = freq[mask]
        impedance = impedance[mask]

    # normalize by bunch spectrum if bunch length is provided
    if bunch_sigma is not None:
        impedance /= GaussianBunch(charge=1, sigma=bunch_sigma).get_form_factor(freq) # Ohm/m^n

    return (
        freq, # Hz
        impedance # Ohm/m^n, n=(2p) for longitudinal, n=(2p-1) for transverse
    )


def get_wake_from_impedance(
    freq: Real1dArray, # Hz
    impedance: Complex1dArray, # Ohm/m^n
    oversampling_factor: float = 1,
    start_time: float = 0, # s
) -> tuple[Real1dArray, Real1dArray]:

    if not freq[0] == 0:
        raise ValueError('Frequency axis must start at 0 Hz and contain positive frequencies only')

    df = _get_uniform_step(freq)

    # inverse of `get_impedance_from_wake`, which uses an even number of time samples
    num_samples = 2 * ceil((len(freq) - 1) * oversampling_factor)
    dt = 1 / (num_samples * df) # s
    time = start_time + dt * np.arange(num_samples) # s

    # undo start time correction of `get_impedance_from_wake`
    impedance = impedance * np.exp(2j * pi * freq * start_time)

    # convention: positive wake is decelerating <-> positive impedance
    wake = np.fft.irfft(impedance, n=num_samples) / dt # V/C/m^n

    return (
        time, # s
        wake # V/C/m^n, n=(2p) for longitudinal, n=(2p-1) for transverse
    )


def convolve_wake(
    time: Real1dArray, # s
    wake: Real1dArray, # V/C (typically)
    target_bunch_sigma: float, # s
    wake_bunch_sigma: float = 0 # s
) -> tuple[Real1dArray, Real1dArray]:

    dt = _get_uniform_step(time)

    if not target_bunch_sigma > wake_bunch_sigma:
        raise ValueError('Can only increase wake bunch length through convolution')

    bunch = GaussianBunch(
        charge=1,
        sigma=sqrt(target_bunch_sigma**2 - wake_bunch_sigma**2)
    )
    bunch._check_sampling(time_step=dt)
    time_bunch = bunch._get_internal_time(time_step=dt)
    bunch_profile = bunch.get_profile(time_bunch)

    logger.info(f'Convolving wake: input sigma = {wake_bunch_sigma*1e9:.2f} ns ({wake_bunch_sigma*c*1e3:.2f} mm), output sigma = {target_bunch_sigma*1e9:.2f} ns ({target_bunch_sigma*c*1e3:.2f} mm)')

    w_convolved = fftconvolve(wake, bunch_profile, mode='full') * dt
    t_convolved = time[0] + time_bunch[0] + np.arange(len(w_convolved)) * dt

    return (
        t_convolved, # s
        w_convolved # V/C (same unit as wake array)
    )


def _get_gaussian_resonator_wake_potential(
    time: Real1dArray, # s
    omega_r: float | np.ndarray, # rad/s, broadcast against `time`
    q_factor: float | np.ndarray, # broadcast against `time`
    bunch_sigma: float # s
) -> np.ndarray:

    # wake function, convolved with a gaussian bunch, for R/Q = 1 and bunch centre at t = 0. With
    # s = -alpha + i*omega_n and u = (t + s*sigma^2) / (sqrt(2)*sigma), the convolution integral is
    #     W(t) = omega_r * Re[(1 + i*alpha/omega_n) * exp(s*t + s^2*sigma^2/2) * erfc(-u) / 2]
    # Evaluating this directly overflows, since the erfc grows as fast as the exponential decays.
    # With the scaled Faddeeva function w(i*z) = exp(z^2) * erfc(z), it splits into
    #     u.real >= 0 : long-range term - short-range term (w evaluated at i*u)
    #     u.real <  0 : short-range term (w evaluated at -i*u)
    # where the long-range term is the damped oscillation with the complex bunch form factor
    # exp(s^2*sigma^2/2), and the short-range term is proportional to exp(-t^2/(2*sigma^2)).
    alpha = omega_r / (2 * q_factor)
    omega_n = omega_r * np.sqrt(1 - 0.25 / q_factor**2)
    s = -alpha + 1j * omega_n
    prefactor = 1 + 1j * alpha / omega_n

    u = (time + s * bunch_sigma**2) / (sqrt(2) * bunch_sigma)
    broadcast_time = np.broadcast_to(time, u.shape)
    is_after_bunch = u.real >= 0

    # exponential grows without bound before the bunch, where the long-range term does not exist
    response = np.zeros_like(u)
    response[is_after_bunch] = np.exp((s * broadcast_time + s**2 * bunch_sigma**2 / 2)[is_after_bunch])

    # far away from the bunch, only the long-range term remains
    is_short_range = np.abs(broadcast_time) < _SHORT_RANGE_SIGMAS * bunch_sigma
    short_range_term = 0.5 * np.exp(-broadcast_time[is_short_range]**2 / (2 * bunch_sigma**2)) * wofz(
        1j * np.where(is_after_bunch, u, -u)[is_short_range]
    )
    response[is_short_range] += np.where(
        is_after_bunch[is_short_range], -short_range_term, short_range_term
    )

    return omega_r * (prefactor * response).real # V/C per Ohm of R/Q


class _Resonator:

    def __init__(
        self,
        resonant_freq: float, # Hz
        r_shunt: float, # Ohm, Ohm/m, ...
        q_factor: float # Q
    ) -> None:
        
        self._freq_r = resonant_freq
        self._r_shunt = r_shunt

        if not q_factor > 0.5:
            raise ValueError(f'For valid resonator model, quality factor must be greater than 0.5, but instead got {q_factor:.2f}')
        self._q_factor = q_factor

    @property
    def resonant_freq(self) -> float:
        return self._freq_r # Hz
    
    @property
    def omega_r(self) -> float:
        return 2 * pi * self._freq_r # rad/s
    
    @property
    def r_shunt(self) -> float:
        return self._r_shunt # Ohm, Ohm/m, ...
    
    @property
    def q_factor(self) -> float:
        return self._q_factor # dimensionless
    
    @property
    def r_over_q(self) -> float:
        return self.r_shunt / self.q_factor # same unit as shunt impedance
    
    @property
    def damping_coefficient(self) -> float:
        return self.omega_r / 2 / self.q_factor # 1/s

    @property
    def reduced_q_factor(self) -> float:
        return sqrt(self.q_factor**2 - 0.25) # dimensionless

    @property
    def reduced_omega_r(self) -> float:
        return self.omega_r * sqrt(1 - 0.25 / self.q_factor**2) # rad/s


class LongitudinalResonator(_Resonator):
    
    def get_loss_factor(
        self,
        bunch_sigma: float = 0 # s
    ) -> float:
        
        # Zotter 5.1.1
        if self.omega_r * bunch_sigma > _MAX_OMEGA_SIGMA_SHORT_BUNCH:
            warnings.warn(
                f'Loss factor calculation valid for short bunches with omega_r * bunch_sigma << 1, but got {self.omega_r * bunch_sigma:.2f}',
                AccuracyWarning, stacklevel=2
            )

        return self.omega_r / 2 * self.r_over_q * (
            1 - 2 / pi * self.omega_r * bunch_sigma / self.q_factor
        ) # V/C
        
    def get_impedance(
        self,
        freq: Real1dArray,
        wake_length: float | None = None # s
    ) -> Complex1dArray:

        if any(freq < 0):
            raise NotImplementedError('Impedance calculation implemented for positive frequencies only')

        omega = 2 * pi * freq # rad/s
        mask = omega > 0 # mask out DC part
        impedance = np.zeros_like(omega, dtype=complex)

        if wake_length is None: # fully decayed wake
            impedance[mask] = self.r_shunt / (
                1 + 1j * self.q_factor * (
                    omega[mask] / self.omega_r - self.omega_r / omega[mask]
                )
            )

        else: # finite wake length, partially decayed
            exp_term = np.exp(-(self.damping_coefficient - 1j * (self.reduced_omega_r - omega[mask])) * wake_length)
            impedance[mask] = (
                (self.r_over_q * omega[mask] / 2 * (1 - exp_term))
                / (self.damping_coefficient - 1j * (self.reduced_omega_r - omega[mask]))
            )

        return impedance # Ohm (or whatever unit of shunt impedance was)
    
    def get_wake_function(
        self,
        time: Real1dArray,
        zero_time_tolerance: float = 1e-15 # s
    ) -> Real1dArray:
        
        wake_function = (
            self.omega_r * self.r_over_q
            * np.exp(-self.damping_coefficient * time)
            * (
                np.cos(self.reduced_omega_r * time)
                - np.sin(self.reduced_omega_r * time) / 2 / self.reduced_q_factor
            )
        )

        # causal wake function
        wake_function[time <= -zero_time_tolerance] = 0 
        # fundamental beam-loading theorem
        wake_function[np.abs(time) < zero_time_tolerance] /= 2

        return wake_function # V/C (or whatever unit of shunt impedance was)

    def get_wake_potential(
        self,
        time: Real1dArray, # s, bunch centre at t = 0
        bunch_sigma: float # s, gaussian
    ) -> Real1dArray:

        if not bunch_sigma > 0:
            raise ValueError(f'Bunch length sigma must be positive, got {bunch_sigma}')

        return self.r_over_q * _get_gaussian_resonator_wake_potential(
            time, self.omega_r, self.q_factor, bunch_sigma
        ) # V/C (or whatever unit of shunt impedance was)


class TransverseResonator(_Resonator):
    
    def get_impedance(
        self,
        freq: Real1dArray,
        wake_length: float | None = None # s
    ) -> Complex1dArray:

        if any(freq < 0):
            raise NotImplementedError('Impedance calculation implemented for positive frequencies only')
        impedance = np.zeros_like(freq, dtype=complex)
        mask = freq > 0 # mask out DC part

        if wake_length is not None:
            raise NotImplementedError('Partially decayed wake not yet implemented for transverse resonator')

        impedance[mask] = self.resonant_freq / freq[mask] * self.r_shunt / (
            1 + 1j * self.q_factor * (
                freq[mask] / self.resonant_freq
                - self.resonant_freq / freq[mask]
            )
        )
        impedance[~mask] = 1j * self.r_over_q

        return impedance # Ohm (or whatever unit of shunt impedance was)

    def get_wake_function(
        self,
        time: Real1dArray,
        zero_time_tolerance: float = 1e-15 # s
    ) -> Real1dArray:

        wake_function = (
            self.omega_r * self.r_over_q * self.omega_r / self.reduced_omega_r
            * np.exp(-self.damping_coefficient * time)
            * np.sin(self.reduced_omega_r * time)
        )

        # causal wake function
        wake_function[time <= -zero_time_tolerance] = 0 

        return wake_function # V/C/m (or whatever unit of shunt impedance was)