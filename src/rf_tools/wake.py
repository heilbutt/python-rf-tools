from math import ceil, pi, sqrt
import numpy as np
from scipy.signal import fftconvolve
from scipy.constants import c
import warnings

from .utils import Real1dArray, Complex1dArray, AccuracyWarning, _get_uniform_step
from .beam import GaussianBunch

import logging
logger = logging.getLogger(__name__) # rf_tools.wake

__all__ = [
    'get_impedance_from_wake',
    'get_wake_from_impedance',
    'convolve_wake',
    'LongitudinalResonator',
    'TransverseResonator'
]

# threshold for `AccuracyWarning`: validity of the short-bunch loss factor approximation
_MAX_OMEGA_SIGMA_SHORT_BUNCH = 0.1


def get_impedance_from_wake(
    time_array: Real1dArray, # s
    wake_array: Real1dArray, # V/C/m^n, n=(2p) for longitudinal, n=(2p-1) for transverse
    oversampling_factor: float = 1,
    bunch_sigma: float | None = None, # s
    cutoff_by_bunch_sigma: float | None = None,
) -> tuple[Real1dArray, Complex1dArray]:

    dt = _get_uniform_step(time_array)

    # even number of samples, so that `get_wake_from_impedance` is the exact inverse
    num_samples = 2 * ceil(len(time_array) * oversampling_factor / 2)
    frequency_array = np.fft.rfftfreq(num_samples, d=dt) # Hz

    # convention: positive wake is decelerating <-> positive impedance
    impedance_array = np.fft.rfft(wake_array, n=num_samples) * dt # Ohm/m^n

    # correct for non-zero start time
    impedance_array *= np.exp(-2j * pi * frequency_array * time_array[0])

    # cutoff before normalization, avoids division by the vanishing bunch spectrum
    if cutoff_by_bunch_sigma is not None:
        if bunch_sigma is None:
            raise ValueError('Cannot cutoff by bunch sigma if `bunch_sigma` is not provided')

        # in angular frequency, the gaussian bunch spectrum has rms width 1/sigma
        cutoff_frequency = cutoff_by_bunch_sigma / (2 * pi * bunch_sigma) # Hz
        mask = frequency_array <= cutoff_frequency
        frequency_array = frequency_array[mask]
        impedance_array = impedance_array[mask]

    # normalize by bunch spectrum if bunch length is provided
    if bunch_sigma is not None:
        impedance_array /= GaussianBunch(charge=1, sigma=bunch_sigma).get_form_factor(frequency_array) # Ohm/m^n

    return (
        frequency_array, # Hz
        impedance_array # Ohm/m^n, n=(2p) for longitudinal, n=(2p-1) for transverse
    )


def get_wake_from_impedance(
    frequency_array: Real1dArray, # Hz
    impedance_array: Complex1dArray, # Ohm/m^n
    oversampling_factor: float = 1,
    start_time: float = 0, # s
) -> tuple[Real1dArray, Real1dArray]:

    if not frequency_array[0] == 0:
        raise ValueError('Frequency axis must start at 0 Hz and contain positive frequencies only')

    df = _get_uniform_step(frequency_array)

    # inverse of `get_impedance_from_wake`, which uses an even number of time samples
    num_samples = 2 * ceil((len(frequency_array) - 1) * oversampling_factor)
    dt = 1 / (num_samples * df) # s
    time_array = start_time + dt * np.arange(num_samples) # s

    # undo start time correction of `get_impedance_from_wake`
    impedance_array = impedance_array * np.exp(2j * pi * frequency_array * start_time)

    # convention: positive wake is decelerating <-> positive impedance
    wake_array = np.fft.irfft(impedance_array, n=num_samples) / dt # V/C/m^n

    return (
        time_array, # s
        wake_array # V/C/m^n, n=(2p) for longitudinal, n=(2p-1) for transverse
    )


def convolve_wake(
    time_array: Real1dArray, # s
    wake_array: Real1dArray, # V/C (typically)
    target_bunch_sigma: float, # s
    wake_bunch_sigma: float = 0 # s
) -> tuple[Real1dArray, Real1dArray]:

    dt = _get_uniform_step(time_array)

    if not target_bunch_sigma > wake_bunch_sigma:
        raise ValueError('Can only increase wake bunch length through convolution')

    bunch = GaussianBunch(
        charge=1,
        sigma=sqrt(target_bunch_sigma**2 - wake_bunch_sigma**2)
    )
    bunch._check_sampling(time_step=dt)
    bunch_time_array = bunch._get_internal_time_array(time_step=dt)
    bunch_profile = bunch.get_profile(bunch_time_array)

    logger.info(f'Convolving wake: input sigma = {wake_bunch_sigma*1e9:.2f} ns ({wake_bunch_sigma*c*1e3:.2f} mm), output sigma = {target_bunch_sigma*1e9:.2f} ns ({target_bunch_sigma*c*1e3:.2f} mm)')

    w_convolved = fftconvolve(wake_array, bunch_profile, mode='full') * dt
    t_convolved = time_array[0] + bunch_time_array[0] + np.arange(len(w_convolved)) * dt

    return (
        t_convolved, # s
        w_convolved # V/C (same unit as wake array)
    )


class _Resonator:

    def __init__(
        self,
        resonant_frequency: float, # Hz
        shunt_impedance: float, # Ohm, Ohm/m, ...
        quality_factor: float # Q
    ) -> None:
        
        self._f_r = resonant_frequency
        self._R = shunt_impedance

        if not quality_factor > 0.5:
            raise ValueError(f'For valid resonator model, quality factor must be greater than 0.5, but instead got {quality_factor:.2f}')
        self._Q = quality_factor

    @property
    def resonant_frequency(self) -> float:
        return self._f_r # Hz
    
    @property
    def omega_r(self) -> float:
        return 2 * pi * self._f_r # rad/s
    
    @property
    def shunt_impedance(self) -> float:
        return self._R # Ohm, Ohm/m, ...
    
    @property
    def quality_factor(self) -> float:
        return self._Q # dimensionless
    
    @property
    def R_over_Q(self) -> float:
        return self.shunt_impedance / self.quality_factor # same unit as shunt impedance
    
    @property
    def damping_coefficient(self) -> float:
        return self.omega_r / 2 / self.quality_factor # 1/s

    @property
    def reduced_quality_factor(self) -> float:
        return sqrt(self.quality_factor**2 - 0.25) # dimensionless

    @property
    def reduced_omega_r(self) -> float:
        return self.omega_r * sqrt(1 - 0.25 / self.quality_factor**2) # rad/s


class LongitudinalResonator(_Resonator):
    
    def get_loss_factor(
        self,
        bunch_length_sigma: float = 0 # s
    ) -> float:
        
        # Zotter 5.1.1
        if self.omega_r * bunch_length_sigma > _MAX_OMEGA_SIGMA_SHORT_BUNCH:
            warnings.warn(
                f'Loss factor calculation valid for short bunches with omega_r * bunch_sigma << 1, but got {self.omega_r * bunch_length_sigma:.2f}',
                AccuracyWarning, stacklevel=2
            )

        return self.omega_r / 2 * self.R_over_Q * (
            1 - 2 / pi * self.omega_r * bunch_length_sigma / self.quality_factor
        ) # V/C
        
    def get_impedance(
        self,
        frequency_array: Real1dArray,
        wake_length: float | None = None # s
    ) -> Complex1dArray:

        if any(frequency_array < 0):
            raise NotImplementedError('Impedance calculation implemented for positive frequencies only')

        omega = 2 * pi * frequency_array # rad/s
        mask = omega > 0 # mask out DC part
        impedance = np.zeros_like(omega, dtype=complex)

        if wake_length is None: # fully decayed wake
            impedance[mask] = self.shunt_impedance / (
                1 + 1j * self.quality_factor * (
                    omega[mask] / self.omega_r - self.omega_r / omega[mask]
                )
            )

        else: # finite wake length, partially decayed
            exp_term = np.exp(-(self.damping_coefficient - 1j * (self.reduced_omega_r - omega[mask])) * wake_length)
            impedance[mask] = (
                (self.R_over_Q * omega[mask] / 2 * (1 - exp_term))
                / (self.damping_coefficient - 1j * (self.reduced_omega_r - omega[mask]))
            )

        return impedance # Ohm (or whatever unit of shunt impedance was)
    
    def get_wake_function(
        self,
        time_array: Real1dArray,
        zero_time_tolerance: float = 1e-15 # s
    ) -> Real1dArray:
        
        wake_function = (
            self.omega_r * self.R_over_Q
            * np.exp(-self.damping_coefficient * time_array)
            * (
                np.cos(self.reduced_omega_r * time_array)
                - np.sin(self.reduced_omega_r * time_array) / 2 / self.reduced_quality_factor
            )
        )

        # causal wake function
        wake_function[time_array <= -zero_time_tolerance] = 0 
        # fundamental beam-loading theorem
        wake_function[np.abs(time_array) < zero_time_tolerance] /= 2

        return wake_function # V/C (or whatever unit of shunt impedance was)
    

class TransverseResonator(_Resonator):
    
    def get_impedance(
        self,
        frequency_array: Real1dArray,
        wake_length: float | None = None # s
    ) -> Complex1dArray:

        if any(frequency_array < 0):
            raise NotImplementedError('Impedance calculation implemented for positive frequencies only')
        impedance = np.zeros_like(frequency_array, dtype=complex)
        mask = frequency_array > 0 # mask out DC part

        if wake_length is not None:
            raise NotImplementedError('Partially decayed wake not yet implemented for transverse resonator')

        impedance[mask] = self.resonant_frequency / frequency_array[mask] * self.shunt_impedance / (
            1 + 1j * self.quality_factor * (
                frequency_array[mask] / self.resonant_frequency
                - self.resonant_frequency / frequency_array[mask]
            )
        )
        impedance[~mask] = 1j * self.R_over_Q

        return impedance # Ohm (or whatever unit of shunt impedance was)

    def get_wake_function(
        self,
        time_array: Real1dArray,
        zero_time_tolerance: float = 1e-15 # s
    ) -> Real1dArray:

        wake_function = (
            self.omega_r * self.R_over_Q * self.omega_r / self.reduced_omega_r
            * np.exp(-self.damping_coefficient * time_array)
            * np.sin(self.reduced_omega_r * time_array)
        )

        # causal wake function
        wake_function[time_array <= -zero_time_tolerance] = 0 

        return wake_function # V/C/m (or whatever unit of shunt impedance was)