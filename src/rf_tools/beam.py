import logging
import warnings
from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from math import ceil, floor, gamma, pi, sqrt

import numpy as np
from scipy.constants import c
from scipy.fft import irfft, next_fast_len, rfft
from scipy.special import betainc, betaincinv, erfc, erfcinv, jv

from .units import format_quantity

logger = logging.getLogger(__name__) # rf_tools.beams

from collections.abc import Sequence
from typing import Literal

from .utils import AccuracyWarning, Complex1dArray, Real1dArray, _get_uniform_step

__all__ = [
    'Beam',
    'BinomialBunch',
    'Bunch',
    'GaussianBunch',
    'NegativeImpedanceHandling'
]

NegativeImpedanceHandling: type = Literal['clip', 'abs', 'raise', 'ignore']

# thresholds for `AccuracyWarning`, relative to the bunch spectrum at DC and the bunch charge
_MAX_FORM_FACTOR = 1e-3 # neglected bunch spectrum above Nyquist or highest impedance frequency
_MAX_TRUNCATED_CHARGE = 1e-4 # bunch charge outside of time window
# internally generated time windows are cheap to extend, so they truncate much less
_MAX_TRUNCATED_CHARGE_INTERNAL = 1e-9


@dataclass(frozen=True) # immutable, so that a `Beam` can safely keep a reference
class Bunch(ABC):

    charge: float # C
    sigma: float # s

    def __post_init__(self) -> None:
        if not self.sigma > 0:
            raise ValueError(f'Bunch length sigma must be positive, got {self.sigma}')

    @abstractmethod
    def _get_normalized_profile(
        self,
        time_array: Real1dArray # s
    ) -> Real1dArray:
        """Bunch profile centred at t = 0, normalized to unit integral."""

    @abstractmethod
    def get_form_factor(
        self,
        frequency_array: Real1dArray # Hz
    ) -> Real1dArray:
        """Analytic bunch spectrum for a bunch centred at t = 0, normalized to unity at DC."""

    @abstractmethod
    def get_truncated_charge_fraction(
        self,
        half_window: float # s
    ) -> float:
        """Fraction of bunch charge outside of +/- `half_window`."""

    @abstractmethod
    def get_half_width(
        self,
        max_truncated_charge_fraction: float
    ) -> float:
        """Half width of the time window outside of which the given fraction of charge is located."""

    def get_profile(
        self,
        time_array: Real1dArray # s
    ) -> Real1dArray:

        return self.charge * self._get_normalized_profile(time_array) # A

    def _get_form_factor_envelope(
        self,
        frequency: float # Hz
    ) -> float:

        # largest form factor at and above `frequency`, assuming monotonic decay
        return float(np.abs(self.get_form_factor(np.array([frequency]))[0]))

    def _get_internal_time_array(
        self,
        time_step: float # s
    ) -> Real1dArray:

        # symmetric time axis that contains t = 0 and covers the entire bunch
        num_half = ceil(self.get_half_width(_MAX_TRUNCATED_CHARGE_INTERNAL) / time_step)
        return time_step * np.arange(-num_half, num_half + 1) # s

    def _check_sampling(
        self,
        time_step: float, # s
        half_window: float | None = None, # s
        stacklevel: int = 3
    ) -> None:

        # bunch spectrum above the Nyquist frequency is aliased
        nyquist_frequency = 1 / (2 * time_step)
        form_factor = self._get_form_factor_envelope(nyquist_frequency)
        if form_factor > _MAX_FORM_FACTOR:
            warnings.warn(
                f'Time step {format_quantity(time_step, "s")} is too coarse for {type(self).__name__} with sigma {format_quantity(self.sigma, "s")} ({self.sigma / time_step:.2f} samples per sigma): '
                f'bunch spectrum at Nyquist frequency {format_quantity(nyquist_frequency, "Hz")} is {form_factor:.1e} of DC and will be aliased. Reduce the time step (increase the maximum frequency)',
                AccuracyWarning, stacklevel=stacklevel
            )

        # charge outside of the time window is missing
        if half_window is not None:
            truncated_fraction = self.get_truncated_charge_fraction(half_window)
            if truncated_fraction > _MAX_TRUNCATED_CHARGE:
                warnings.warn(
                    f'Time window +/- {format_quantity(half_window, "s")} ({half_window / self.sigma:.2f} sigma) truncates {truncated_fraction:.1e} of the charge of {type(self).__name__} with sigma {format_quantity(self.sigma, "s")}',
                    AccuracyWarning, stacklevel=stacklevel
                )

    def get_numeric_spectrum(
        self,
        max_frequency: float, # Hz
        frequency_step: float # Hz
    ) -> tuple[Real1dArray, Complex1dArray]:

        # FFT of the sampled profile, reference for `charge * get_form_factor`
        time_period = 1 / frequency_step
        time_step = 1 / (2 * max_frequency)
        self._check_sampling(time_step=time_step, half_window=time_period/2)

        num_samples = round(time_period / time_step)
        time_array = np.linspace(
            start=-time_period/2, stop=time_period/2, num=num_samples, endpoint=False
        )

        profile = self.get_profile(time_array)
        spectrum_array = np.fft.rfft(profile) * time_step
        frequency_array = np.fft.rfftfreq(len(profile), time_step)

        # correct for non-zero start time, so that t = 0 is at the bunch centre
        spectrum_array *= np.exp(-2j * pi * frequency_array * time_array[0])

        return (
            frequency_array, # Hz
            spectrum_array # C
        )

    def get_loss_factor(
        self,
        time_array: Real1dArray, # s
        wake_array: Real1dArray, # V/C/m^p
    ) -> float:

        dt = _get_uniform_step(time_array)
        self._check_sampling(time_step=dt)

        # extend wake time axis with zeros to cover the entire bunch
        extent = self.get_half_width(_MAX_TRUNCATED_CHARGE_INTERNAL) # s
        num_before = max(0, ceil((time_array[0] + extent) / dt))
        num_after = max(0, ceil((extent - time_array[-1]) / dt))
        time_array = time_array[0] + dt * np.arange(
            -num_before, len(time_array) + num_after
        ) # s
        wake_potential = np.pad(wake_array, (num_before, num_after)) # V/C

        profile = self._get_normalized_profile(time_array) # 1/s

        # convention: positive wake is decelerating <-> positive loss factor
        loss_factor = float(np.trapezoid(profile * wake_potential, time_array))

        return loss_factor # V/C/m^p, same unit as wake


@dataclass(frozen=True)
class GaussianBunch(Bunch):

    def _get_normalized_profile(
        self,
        time_array: Real1dArray # s
    ) -> Real1dArray:

        amplitude = 1 / (self.sigma * sqrt(2 * pi))
        return amplitude * np.exp(-0.5 * (time_array / self.sigma) ** 2) # 1/s

    def get_form_factor(
        self,
        frequency_array: Real1dArray # Hz
    ) -> Real1dArray:

        return np.exp(-0.5 * (2 * pi * frequency_array * self.sigma)**2)

    def get_truncated_charge_fraction(
        self,
        half_window: float # s
    ) -> float:

        return float(erfc(half_window / (sqrt(2) * self.sigma)))

    def get_half_width(
        self,
        max_truncated_charge_fraction: float
    ) -> float:

        return sqrt(2) * self.sigma * float(erfcinv(max_truncated_charge_fraction)) # s


@dataclass(frozen=True)
class BinomialBunch(Bunch):

    exponent: float # mu, 0 for uniform, -> infinity for gaussian distribution

    def __post_init__(self) -> None:
        super().__post_init__()
        if not self.exponent >= 0:
            raise ValueError(f'Binomial exponent (mu) must be non-negative, got {self.exponent}')

    @property
    def full_bunch_length(self) -> float:
        return 2 * self.sigma * sqrt(3 + 2 * self.exponent) # s

    def _get_normalized_profile(
        self,
        time_array: Real1dArray # s
    ) -> Real1dArray:

        amplitude = (
            2 * gamma(1.5 + self.exponent)
            / (self.full_bunch_length * sqrt(pi) * gamma(1 + self.exponent))
        )
        profile = np.clip(1 - 4 * (time_array / self.full_bunch_length) ** 2, min=0, max=None) ** self.exponent
        profile[np.abs(time_array) > self.full_bunch_length / 2] = 0

        return amplitude * profile # 1/s

    def get_form_factor(
        self,
        frequency_array: Real1dArray # Hz
    ) -> Real1dArray:

        order = self.exponent + 0.5
        x = np.abs(pi * frequency_array * self.full_bunch_length) # omega * full_bunch_length / 2
        form_factor = np.ones_like(x)
        nonzero = x > 0
        form_factor[nonzero] = gamma(order + 1) * (x[nonzero] / 2)**(-order) * jv(order, x[nonzero])

        return form_factor

    def get_truncated_charge_fraction(
        self,
        half_window: float # s
    ) -> float:

        u = min(1, 2 * half_window / self.full_bunch_length)
        return float(1 - betainc(0.5, self.exponent + 1, u**2))

    def get_half_width(
        self,
        max_truncated_charge_fraction: float
    ) -> float:

        return self.full_bunch_length / 2 * sqrt(float(
            betaincinv(0.5, self.exponent + 1, 1 - max_truncated_charge_fraction)
        )) # s

    def _get_form_factor_envelope(
        self,
        frequency: float # Hz
    ) -> float:

        # form factor has zeros, so take the maximum over two oscillation periods above `frequency`
        frequency_array = frequency + np.linspace(0, 2 / self.full_bunch_length, 65)
        return float(np.max(np.abs(self.get_form_factor(frequency_array))))


@dataclass(frozen=True) # immutable, spectra and profiles are recomputed on demand
class Beam:

    circumference: float # m
    max_frequency: float # Hz
    filling_scheme: Sequence[bool] = field(repr=False) # stored as tuple, one entry per bucket
    bunch: Bunch # identical for all filled buckets
    beta: float = 1.0
    harmonic_number_validation: InitVar[int | None] = None
    num_bunches_validation: InitVar[int | None] = None

    def __post_init__(
        self,
        harmonic_number_validation: int | None,
        num_bunches_validation: int | None
    ) -> None:

        # tuple prevents modification of the filling scheme after construction
        object.__setattr__(self, 'filling_scheme', tuple(self.filling_scheme))

        if not any(self.filling_scheme):
            raise ValueError('Filling scheme must contain at least one filled bucket (`True`)')
        if harmonic_number_validation and (harmonic_number_validation != self.harmonic_number):
            raise ValueError('Filling scheme validation: length does not correspond to harmonic number')
        if num_bunches_validation and (num_bunches_validation != self.num_bunches):
            raise ValueError('Filling scheme validation: sum does not correspond to number of bunches')
        if self.beta != 1.0:
            raise NotImplementedError('Only relativistic beams with `beta` = 1 are currently supported')

    @property
    def revolution_frequency(self) -> float:
        return c / self.circumference # Hz
    
    @property
    def revolution_period(self) -> float:
        return self.circumference / c # s

    @property
    def harmonic_number(self) -> int:
        return len(self.filling_scheme)

    @property
    def bucket_length(self) -> float:
        return self.revolution_period / self.harmonic_number # s
    
    @property
    def num_bunches(self) -> int:
        return sum(self.filling_scheme)
    
    @property
    def charge(self) -> float:
        return self.bunch.charge * self.num_bunches # C
    
    @property
    def average_current(self) -> float:
        return self.charge * self.revolution_frequency # A

    def get_profile(self) -> tuple[Real1dArray, Real1dArray]:

        # generate time axis centered for one bucket (t = 0 at bucket center)
        # with sampling such that the concatenated time-axis is equidistant
        delta_t = 1 / (2 * self.max_frequency)
        num_samples_per_bucket = ceil(self.bucket_length / delta_t)
        
        # time axis for one bunch
        bunch_time_array = np.linspace(
            start=-self.bucket_length/2, stop=self.bucket_length/2,
            num=num_samples_per_bucket, endpoint=False
        )
        self.bunch._check_sampling(
            time_step=float(bunch_time_array[1] - bunch_time_array[0]),
            half_window=self.bucket_length / 2
        )

        bunch_profile = self.bunch.get_profile(bunch_time_array) # A

        logger.info(f'Generating beam profile: {self.harmonic_number} buckets ({self.num_bunches} filled), {num_samples_per_bucket} samples/bucket, {self.harmonic_number*num_samples_per_bucket} total samples')

        # one row per bucket, filled buckets contain the bunch profile, empty ones zeros
        bucket_start_times = self.bucket_length * np.arange(self.harmonic_number) # s
        beam_time_array = np.add.outer(bucket_start_times, bunch_time_array).ravel() # s
        beam_profile = np.outer(np.asarray(self.filling_scheme, dtype=float), bunch_profile).ravel() # A

        return (
            beam_time_array, # s
            beam_profile # A
        )

    def _get_harmonic_numbers(self) -> np.ndarray[tuple[int], np.dtype[np.int_]]:

        # revolution harmonics up to the maximum frequency (tolerance for rounding errors)
        max_harmonic = floor(self.max_frequency / self.revolution_frequency * (1 + 1e-9))
        return np.arange(max_harmonic + 1)

    def get_spectrum(self) -> tuple[Real1dArray, Complex1dArray]:

        harmonic_array = self._get_harmonic_numbers()
        frequency_array = harmonic_array * self.revolution_frequency # Hz

        logger.info(f'Computing beam spectrum: {len(frequency_array)} revolution harmonics up to {format_quantity(frequency_array[-1], "Hz")}')

        # bunch in bucket k is delayed by k bucket lengths, i.e. by a phase of -2 pi n k / h at harmonic n,
        # so the filling pattern contributes its discrete Fourier transform, periodic in n with period h
        filling_pattern_spectrum: Complex1dArray = np.fft.fft(np.asarray(self.filling_scheme, dtype=float))
        spectrum_array = (
            self.bunch.charge
            * self.bunch.get_form_factor(frequency_array)
            * filling_pattern_spectrum[harmonic_array % self.harmonic_number]
        ) # C

        return (
            frequency_array, # Hz
            spectrum_array # C
        )

    def _get_squared_spectrum(self) -> tuple[Real1dArray, Real1dArray]:

        # |spectrum|^2 for power calculations, avoids the complex spectrum to save memory
        harmonic_array = self._get_harmonic_numbers()
        frequency_array = harmonic_array * self.revolution_frequency # Hz

        filling_pattern_power = np.abs(np.fft.fft(np.asarray(self.filling_scheme, dtype=float)))**2

        # in-place operations to avoid temporary arrays
        squared_spectrum = self.bunch.get_form_factor(frequency_array)
        squared_spectrum **= 2
        squared_spectrum *= self.bunch.charge**2
        squared_spectrum *= filling_pattern_power[harmonic_array % self.harmonic_number] # C^2

        return (
            frequency_array, # Hz
            squared_spectrum # C^2
        )

    def get_numeric_spectrum(self) -> tuple[Real1dArray, Complex1dArray]:

        # FFT of the sampled beam profile, reference for `get_spectrum`
        beam_time_array, beam_profile = self.get_profile() # A
        time_step = beam_time_array[1] - beam_time_array[0]

        logger.info(f'Computing numeric beam spectrum: {len(beam_profile)} samples, frequency resolution {format_quantity(self.revolution_frequency, "Hz")}')

        spectrum_array = time_step * np.fft.rfft(beam_profile) # C
        frequency_array = np.fft.rfftfreq(len(beam_profile), time_step) # Hz

        # correct for non-zero start time, so that t = 0 is at the centre of the first bucket
        spectrum_array *= np.exp(-2j * pi * frequency_array * beam_time_array[0])

        return (
            frequency_array, # Hz
            spectrum_array # C
        )

    def _get_real_impedance_at_beam_frequencies(
        self,
        beam_freq: Real1dArray, # Hz
        frequency_array: Real1dArray, # Hz
        impedance_array: Complex1dArray, # Ohm
        negative_real_impedance_handling: NegativeImpedanceHandling
    ) -> Real1dArray:

        max_impedance_delta_f = float(np.max(np.diff(frequency_array)))
        if (max_impedance_delta_f > self.revolution_frequency):
            warnings.warn(
                f'Impedance frequency resolution {format_quantity(max_impedance_delta_f, "Hz")} is coarser than revolution frequency {format_quantity(self.revolution_frequency, "Hz")}, narrow-band features may be missed',
                AccuracyWarning, stacklevel=3
            )

        # power is only summed up to the highest revolution harmonic of the beam spectrum,
        # and the impedance is assumed zero above its highest frequency
        max_frequency = min(float(beam_freq[-1]), float(frequency_array[-1])) # Hz
        form_factor = self.bunch._get_form_factor_envelope(max_frequency)
        if form_factor > _MAX_FORM_FACTOR:
            limited_by = 'maximum frequency of the beam' if beam_freq[-1] < frequency_array[-1] else 'highest impedance frequency'
            warnings.warn(
                f'Power loss only summed up to {format_quantity(max_frequency, "Hz")} ({limited_by}), where the bunch spectrum is still {form_factor:.1e} of DC. Power loss at higher frequencies is neglected',
                AccuracyWarning, stacklevel=3
            )

        real_impedance_at_beam_freq = np.interp(
            x=beam_freq,
            xp=frequency_array,
            fp=np.real(impedance_array),
            left=0.0, right=0.0
        ) # Ohm

        if negative_real_impedance_handling == 'clip':
            real_impedance_at_beam_freq = np.clip(real_impedance_at_beam_freq, min=0, max=None)
        elif negative_real_impedance_handling == 'abs':
            real_impedance_at_beam_freq = np.abs(real_impedance_at_beam_freq)
        elif negative_real_impedance_handling == 'raise':
            num_negative = int(np.sum(real_impedance_at_beam_freq < 0))
            if num_negative > 0:
                raise ValueError(f'Real part of impedance is negative at {num_negative} ({100*num_negative/len(real_impedance_at_beam_freq):.1f} %) of the beam spectral lines')
        elif negative_real_impedance_handling == 'ignore':
            pass
        else:
            raise ValueError(f'Unknown handling of negative impedance real part: `{negative_real_impedance_handling}`')

        return real_impedance_at_beam_freq # Ohm

    def get_power_loss_spectrum(
        self,
        frequency_array: Real1dArray, # Hz
        impedance_array: Complex1dArray, # Ohm
        negative_real_impedance_handling: NegativeImpedanceHandling = 'raise'
    ) -> tuple[Real1dArray, Real1dArray]:

        beam_freq, squared_beam_spectrum = self._get_squared_spectrum() # Hz, C^2
        real_impedance_at_beam_freq = self._get_real_impedance_at_beam_frequencies(
            beam_freq=beam_freq,
            frequency_array=frequency_array,
            impedance_array=impedance_array,
            negative_real_impedance_handling=negative_real_impedance_handling
        ) # Ohm

        power_loss_spectrum = (
            self.revolution_frequency**2
            * squared_beam_spectrum * real_impedance_at_beam_freq
        ) # Hz^2 * C^2 * Ohm = W

        return (
            beam_freq, # Hz
            power_loss_spectrum # W
        )

    def get_power_loss(
        self,
        frequency_array: Real1dArray, # Hz
        impedance_array: Complex1dArray, # Ohm
        negative_real_impedance_handling: NegativeImpedanceHandling = 'raise',
    ) -> float:

        _, power_loss_spectrum = self.get_power_loss_spectrum(
            frequency_array=frequency_array,
            impedance_array=impedance_array,
            negative_real_impedance_handling=negative_real_impedance_handling
        ) # W
        # positive and negative frequencies contribute equally, except for DC
        power_loss = 2 * float(np.sum(power_loss_spectrum)) - float(power_loss_spectrum[0]) # W

        return power_loss # W

    def get_shifted_power_loss(
        self,
        frequency_array: Real1dArray, # Hz
        value_array: Complex1dArray, # Ohm
        max_frequency_shift: float, # Hz
        negative_real_impedance_handling: NegativeImpedanceHandling = 'raise'
    ) -> tuple[Real1dArray, Real1dArray]:

        beam_freq, squared_beam_spectrum = self._get_squared_spectrum() # Hz, C^2
        real_impedance_at_beam_freq = self._get_real_impedance_at_beam_frequencies(
            beam_freq=beam_freq,
            frequency_array=frequency_array,
            impedance_array=value_array,
            negative_real_impedance_handling=negative_real_impedance_handling
        ) # Ohm

        frequency_step = beam_freq[1] - beam_freq[0]
        max_step = ceil(max_frequency_shift / frequency_step)
        shift_steps = np.arange(-max_step, max_step + 1)
        num_harmonics = len(squared_beam_spectrum)

        logger.info(f'Calculating shifted power losses for frequency shifts +/- {format_quantity(max_frequency_shift, "Hz")} in steps of {format_quantity(frequency_step, "Hz")} ({len(shift_steps)} steps)')

        # impedance shifted by s harmonics is Z[n - s] at harmonic n, zero outside of the array, so the
        # power loss is a cross-correlation: sum_n S[n] Z[n - s], computed for all shifts at once via FFT.
        # Circular correlation of length >= num_harmonics + max_step is exact for |s| <= max_step
        fft_length = next_fast_len(num_harmonics + max_step, real=True)
        correlation_spectrum = rfft(squared_beam_spectrum, n=fft_length)
        correlation_spectrum *= np.conj(rfft(real_impedance_at_beam_freq, n=fft_length))
        correlation = irfft(correlation_spectrum, n=fft_length) # C^2 * Ohm, shift s at index s mod fft_length

        # positive and negative frequencies contribute equally, except for DC,
        # where the shifted impedance is Z[-s] for s <= 0 and zero otherwise
        dc_impedance = np.where(
            (shift_steps <= 0) & (-shift_steps < num_harmonics),
            real_impedance_at_beam_freq[np.clip(-shift_steps, 0, num_harmonics - 1)],
            0.0
        ) # Ohm
        power_losses = self.revolution_frequency**2 * (
            2 * correlation[shift_steps % fft_length]
            - squared_beam_spectrum[0] * dc_impedance
        ) # Hz^2 * C^2 * Ohm = W

        logger.info('Power loss calculation results:')
        logger.info(f'  max:    {format_quantity(max(power_losses), "W"):>12} at shift {format_quantity(shift_steps[np.argmax(power_losses)] * frequency_step, "Hz"):>12}')
        logger.info(f'  min:    {format_quantity(min(power_losses), "W"):>12} at shift {format_quantity(shift_steps[np.argmin(power_losses)] * frequency_step, "Hz"):>12}')
        logger.info(f'  mean:   {format_quantity(float(np.mean(power_losses)), "W"):>12}')
        logger.info(f'  median: {format_quantity(float(np.median(power_losses)), "W"):>12}')
        
        return (
            frequency_step * shift_steps, # Hz
            power_losses # W
        )
