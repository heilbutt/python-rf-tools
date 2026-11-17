import logging
import warnings
from dataclasses import dataclass
from math import ceil, pi, sqrt

import numpy as np
from scipy.constants import c
from scipy.signal import fftconvolve

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


def get_impedance_from_wake(
    t: Real1dArray, # s
    w: Real1dArray, # V/C/m^n, n=(2p) for longitudinal, n=(2p-1) for transverse
    oversampling: float = 1,
    sigma: float | None = None, # s
    cutoff_sigma: float | None = None,
) -> tuple[Real1dArray, Complex1dArray]:

    dt = _get_uniform_step(t)

    # even number of samples, so that `get_wake_from_impedance` is the exact inverse
    num_samples = 2 * ceil(len(t) * oversampling / 2)
    f = np.fft.rfftfreq(num_samples, d=dt) # Hz

    # convention: positive wake is decelerating <-> positive impedance
    z = np.fft.rfft(w, n=num_samples) * dt # Ohm/m^n

    # correct for non-zero start time
    z *= np.exp(-2j * pi * f * t[0])

    # cutoff before normalization, avoids division by the vanishing bunch spectrum
    if cutoff_sigma is not None:
        if sigma is None:
            raise ValueError('Cannot cutoff by bunch sigma if `sigma` is not provided')

        # in angular frequency, the gaussian bunch spectrum has rms width 1/sigma
        f_cutoff = cutoff_sigma / (2 * pi * sigma) # Hz
        mask = f <= f_cutoff
        f = f[mask]
        z = z[mask]

    # normalize by bunch spectrum if bunch length is provided
    if sigma is not None:
        z /= GaussianBunch(charge=1, sigma=sigma).get_form_factor(f) # Ohm/m^n

    return (
        f, # Hz
        z # Ohm/m^n, n=(2p) for longitudinal, n=(2p-1) for transverse
    )


def get_wake_from_impedance(
    f: Real1dArray, # Hz
    z: Complex1dArray, # Ohm/m^n
    oversampling: float = 1,
    t_start: float = 0, # s
) -> tuple[Real1dArray, Real1dArray]:

    if not f[0] == 0:
        raise ValueError('Frequency axis must start at 0 Hz and contain positive frequencies only')

    df = _get_uniform_step(f)

    # inverse of `get_impedance_from_wake`, which uses an even number of time samples
    num_samples = 2 * ceil((len(f) - 1) * oversampling)
    dt = 1 / (num_samples * df) # s
    t = t_start + dt * np.arange(num_samples) # s

    # convention: positive wake is decelerating <-> positive impedance
    # undo start time correction of `get_impedance_from_wake`
    w = np.fft.irfft(
        z * np.exp(2j * pi * f * t_start), n=num_samples
    ) / dt # V/C/m^n

    return (
        t, # s
        w # V/C/m^n, n=(2p) for longitudinal, n=(2p-1) for transverse
    )


def convolve_wake(
    t: Real1dArray, # s
    w: Real1dArray, # V/C (typically)
    target_sigma: float, # s
    wake_sigma: float = 0 # s
) -> tuple[Real1dArray, Real1dArray]:

    dt = _get_uniform_step(t)

    if not target_sigma > wake_sigma:
        raise ValueError('Can only increase wake bunch length through convolution')

    bunch = GaussianBunch(
        charge=1,
        sigma=sqrt(target_sigma**2 - wake_sigma**2)
    )
    bunch._check_sampling(time_step=dt)
    t_bunch = bunch._get_internal_time_array(time_step=dt)
    bunch_profile = bunch.get_profile(t_bunch)

    logger.info(f'Convolving wake: input sigma = {wake_sigma*1e9:.2f} ns ({wake_sigma*c*1e3:.2f} mm), output sigma = {target_sigma*1e9:.2f} ns ({target_sigma*c*1e3:.2f} mm)')

    w_convolved = fftconvolve(w, bunch_profile, mode='full') * dt
    t_convolved = t[0] + t_bunch[0] + np.arange(len(w_convolved)) * dt

    return (
        t_convolved, # s
        w_convolved # V/C (same unit as wake array)
    )


@dataclass(frozen=True)
class _Resonator:

    f_r: float # Hz
    R: float # Ohm, Ohm/m, ...
    Q: float # dimensionless

    def __post_init__(self) -> None:
        if not self.Q > 0.5:
            raise ValueError(f'For valid resonator model, quality factor must be greater than 0.5, but instead got {self.Q:.2f}')

    @property
    def omega_r(self) -> float:
        return 2 * pi * self.f_r # rad/s

    @property
    def R_over_Q(self) -> float:
        return self.R / self.Q # same unit as shunt impedance

    @property
    def alpha(self) -> float:
        return self.omega_r / 2 / self.Q # 1/s

    @property
    def Q_reduced(self) -> float:
        return sqrt(self.Q**2 - 0.25) # dimensionless

    @property
    def omega_reduced(self) -> float:
        return self.omega_r * sqrt(1 - 0.25 / self.Q**2) # rad/s


class LongitudinalResonator(_Resonator):

    def get_loss_factor(
        self,
        sigma: float = 0 # s
    ) -> float:

        # Zotter 5.1.1
        if self.omega_r * sigma > _MAX_OMEGA_SIGMA_SHORT_BUNCH:
            warnings.warn(
                f'Loss factor calculation valid for short bunches with omega_r * sigma << 1, but got {self.omega_r * sigma:.2f}',
                AccuracyWarning, stacklevel=2
            )

        return self.omega_r / 2 * self.R_over_Q * (
            1 - 2 / pi * self.omega_r * sigma / self.Q
        ) # V/C

    def get_impedance(
        self,
        f: Real1dArray,
        wake_length: float | None = None # s
    ) -> Complex1dArray:

        if any(f < 0):
            raise NotImplementedError('Impedance calculation implemented for positive frequencies only')

        omega = 2 * pi * f # rad/s
        mask = omega > 0 # mask out DC part
        z = np.zeros_like(omega, dtype=complex)

        if wake_length is None: # fully decayed wake
            z[mask] = self.R / (
                1 + 1j * self.Q * (
                    omega[mask] / self.omega_r - self.omega_r / omega[mask]
                )
            )

        else: # finite wake length, partially decayed
            exp_term = np.exp(-(self.alpha - 1j * (self.omega_reduced - omega[mask])) * wake_length)
            z[mask] = (
                (self.R_over_Q * omega[mask] / 2 * (1 - exp_term))
                / (self.alpha - 1j * (self.omega_reduced - omega[mask]))
            )

        return z # Ohm (or whatever unit of shunt impedance was)

    def get_wake_function(
        self,
        t: Real1dArray,
        zero_time_tolerance: float = 1e-15 # s
    ) -> Real1dArray:

        w = (
            self.omega_r * self.R_over_Q
            * np.exp(-self.alpha * t)
            * (
                np.cos(self.omega_reduced * t)
                - np.sin(self.omega_reduced * t) / 2 / self.Q_reduced
            )
        )

        # causal wake function
        w[t <= -zero_time_tolerance] = 0
        # fundamental beam-loading theorem
        w[np.abs(t) < zero_time_tolerance] /= 2

        return w # V/C (or whatever unit of shunt impedance was)


class TransverseResonator(_Resonator):

    def get_impedance(
        self,
        f: Real1dArray,
        wake_length: float | None = None # s
    ) -> Complex1dArray:

        if any(f < 0):
            raise NotImplementedError('Impedance calculation implemented for positive frequencies only')
        z = np.zeros_like(f, dtype=complex)
        mask = f > 0 # mask out DC part

        if wake_length is not None:
            raise NotImplementedError('Partially decayed wake not yet implemented for transverse resonator')

        z[mask] = self.f_r / f[mask] * self.R / (
            1 + 1j * self.Q * (
                f[mask] / self.f_r
                - self.f_r / f[mask]
            )
        )
        z[~mask] = 1j * self.R_over_Q

        return z # Ohm (or whatever unit of shunt impedance was)

    def get_wake_function(
        self,
        t: Real1dArray,
        zero_time_tolerance: float = 1e-15 # s
    ) -> Real1dArray:

        w = (
            self.omega_r * self.R_over_Q * self.omega_r / self.omega_reduced
            * np.exp(-self.alpha * t)
            * np.sin(self.omega_reduced * t)
        )

        # causal wake function
        w[t <= -zero_time_tolerance] = 0

        return w # V/C/m (or whatever unit of shunt impedance was)
