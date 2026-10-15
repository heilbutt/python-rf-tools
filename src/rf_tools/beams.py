import numpy as np
from scipy.constants import c
from scipy.special import betainc, betaincinv, erfc, erfcinv, jv
from math import pi, gamma, sqrt, ceil
from tqdm import tqdm
import warnings

from .units import format_quantity

import logging
logger = logging.getLogger(__name__) # rf_tools.beams

from typing import Literal, Sequence, TypeAlias
from .utils import Real1dArray, Complex1dArray, AccuracyWarning, _get_uniform_step

__all__ = [
    'Bunch',
    'Beam',
    'NegativeImpedanceHandling'
]

NegativeImpedanceHandling: TypeAlias = Literal['clip', 'abs', 'raise', 'ignore']

# thresholds for `AccuracyWarning`, relative to the bunch spectrum at DC and the bunch charge
_MAX_FORM_FACTOR = 1e-3 # neglected bunch spectrum above Nyquist or highest impedance frequency
_MAX_TRUNCATED_CHARGE = 1e-4 # bunch charge outside of time window
# internally generated time windows are cheap to extend, so they truncate much less
_MAX_TRUNCATED_CHARGE_INTERNAL = 1e-9


def _get_binomial_full_bunch_length(
    sigma: float, # s
    exponent: float
) -> float:

    return 2 * sigma * sqrt(3 + 2 * exponent) # s


def _get_binomial_bunch_profile(
    time_array: Real1dArray, # s
    sigma: float, # s
    exponent: float,
) -> Real1dArray:

    full_bunch_length = _get_binomial_full_bunch_length(sigma, exponent)
    amplitude = (
        2 * gamma(1.5 + exponent)
        / (full_bunch_length * sqrt(pi) * gamma(1 + exponent))
    )
    profile = np.clip(1 - 4 * (time_array / full_bunch_length) ** 2, min=0, max=None) ** exponent
    profile[np.abs(time_array) > full_bunch_length / 2] = 0
    
    return amplitude * profile # 1/s


def _get_gaussian_bunch_profile(
    time_array: Real1dArray, # s
    sigma: float # s
) -> Real1dArray:

    amplitude = 1 / (sigma * sqrt(2 * pi))
    profile = np.exp(-0.5 * (time_array / sigma) ** 2)
    
    return amplitude * profile # 1/s


class Bunch:

    def __init__(
        self,
        charge: float, # C
        sigma: float, # s
        distribution: Literal['gaussian', 'binomial'],
        binomial_exponent: float | None = None,
    ) -> None:

        if distribution == 'binomial':
            if binomial_exponent is None:
                raise ValueError('Must specify binomial exponent (mu) for binomial bunch distribution')
        elif distribution == 'gaussian':
            if binomial_exponent is not None:
                raise ValueError('Binomial exponent (mu) should not be specified for gaussian bunch distribution')
        else:
            raise ValueError(f'Unknown bunch distribution `{distribution}`')
        if not sigma > 0:
            raise ValueError(f'Bunch length sigma must be positive, got {sigma}')
        
        self.distribution = distribution
        self.sigma = sigma
        self.charge = charge
        self.binomial_exponent = binomial_exponent

    @property
    def _binomial_parameters(self) -> tuple[float, float]:
        assert self.binomial_exponent is not None
        return (
            _get_binomial_full_bunch_length(self.sigma, self.binomial_exponent), # s
            self.binomial_exponent
        )
    
    def get_profile(
        self,
        time_array: Real1dArray # s
    ) -> Real1dArray:

        # get normalized bunch profile according to distribution, unit is 1/s
        if self.distribution == 'binomial':
            assert self.binomial_exponent is not None
            profile = _get_binomial_bunch_profile(
                time_array=time_array,
                sigma=self.sigma,
                exponent=self.binomial_exponent
            )
        else:
            profile = _get_gaussian_bunch_profile(
                time_array=time_array,
                sigma=self.sigma
            )
        
        return self.charge * profile # A

    def get_form_factor(
        self,
        frequency_array: Real1dArray # Hz
    ) -> Real1dArray:

        # analytic bunch spectrum, normalized to unity at DC, for a bunch centred at t = 0
        if self.distribution == 'gaussian':
            return np.exp(-0.5 * (2 * pi * frequency_array * self.sigma)**2)

        full_bunch_length, exponent = self._binomial_parameters
        order = exponent + 0.5
        x = np.abs(pi * frequency_array * full_bunch_length) # omega * full_bunch_length / 2
        form_factor = np.ones_like(x)
        nonzero = x > 0
        form_factor[nonzero] = gamma(order + 1) * (x[nonzero] / 2)**(-order) * jv(order, x[nonzero])

        return form_factor

    def get_truncated_charge_fraction(
        self,
        half_window: float # s
    ) -> float:

        # fraction of bunch charge outside of +/- half_window
        if self.distribution == 'gaussian':
            return float(erfc(half_window / (sqrt(2) * self.sigma)))

        full_bunch_length, exponent = self._binomial_parameters
        u = min(1, 2 * half_window / full_bunch_length)
        return float(1 - betainc(0.5, exponent + 1, u**2))

    def get_half_width(
        self,
        max_truncated_charge_fraction: float
    ) -> float:

        # half width of the time window outside of which the given fraction of charge is located
        if self.distribution == 'gaussian':
            return sqrt(2) * self.sigma * float(erfcinv(max_truncated_charge_fraction)) # s

        full_bunch_length, exponent = self._binomial_parameters
        return full_bunch_length / 2 * sqrt(float(
            betaincinv(0.5, exponent + 1, 1 - max_truncated_charge_fraction)
        )) # s

    def _get_form_factor_envelope(
        self,
        frequency: float # Hz
    ) -> float:

        # maximum over one oscillation period above `frequency`, as the binomial form factor has zeros
        oscillation_period = 1 / self.get_half_width(_MAX_TRUNCATED_CHARGE_INTERNAL) # Hz
        frequency_array = frequency + np.linspace(0, oscillation_period, 65)
        return float(np.max(np.abs(self.get_form_factor(frequency_array))))

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
                f'Time step {format_quantity(time_step, "s")} is too coarse for {self.distribution} bunch with sigma {format_quantity(self.sigma, "s")} ({self.sigma / time_step:.2f} samples per sigma): '
                f'bunch spectrum at Nyquist frequency {format_quantity(nyquist_frequency, "Hz")} is {form_factor:.1e} of DC and will be aliased. Reduce the time step (increase the maximum frequency)',
                AccuracyWarning, stacklevel=stacklevel
            )

        # charge outside of the time window is missing
        if half_window is not None:
            truncated_fraction = self.get_truncated_charge_fraction(half_window)
            if truncated_fraction > _MAX_TRUNCATED_CHARGE:
                warnings.warn(
                    f'Time window +/- {format_quantity(half_window, "s")} ({half_window / self.sigma:.2f} sigma) truncates {truncated_fraction:.1e} of the charge of {self.distribution} bunch with sigma {format_quantity(self.sigma, "s")}',
                    AccuracyWarning, stacklevel=stacklevel
                )

    def get_spectrum(
        self,
        max_frequency: float, # Hz
        frequency_step: float # Hz
    ) -> tuple[Real1dArray, Complex1dArray]:
        
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

        profile = self.get_profile(time_array) # A
        profile /= self.charge # 1/s

        # convention: positive wake is decelerating <-> positive loss factor
        loss_factor = float(np.trapezoid(profile * wake_potential, time_array))

        return loss_factor # V/C/m^p, same unit as wake


class Beam:

    def __init__(
        self,
        circumference: float, # m
        max_frequency: float, # Hz
        filling_scheme: Sequence[bool],
        bunch_charge: float, # C
        bunch_sigma: float, # s
        bunch_distribution: Literal['gaussian', 'binomial'],
        bunch_binomial_exponent: float | None = None,
        beta: float = 1.0,
        harmonic_number_validation: int | None = None,
        num_bunches_validation: int | None = None
    ) -> None:
        
        self._circumference = circumference
        self._max_frequency = max_frequency
        
        if not any(filling_scheme):
            raise ValueError('Filling scheme must contain at least one filled bucket (`True`)')
        if harmonic_number_validation and (harmonic_number_validation != len(filling_scheme)):
            raise ValueError('Filling scheme validation: length does not correspond to harmonic number')
        if num_bunches_validation and (num_bunches_validation != sum(filling_scheme)):
            raise ValueError('Filling scheme validation: sum does not correspond to number of bunches')
        self._filling_scheme = list(filling_scheme) # creates copy to avoid accidental modification
        
        self._bunch_charge = bunch_charge
        self._bunch_sigma = bunch_sigma
        self._bunch_distribution = bunch_distribution
        self._bunch_binomial_exponent = bunch_binomial_exponent
        if beta != 1.0:
            raise NotImplementedError('Only relativistic beams with `beta` = 1 are currently supported')

        # validates distribution parameters and sampling before any expensive calculation
        self._unit_charge_bunch = Bunch(
            charge=1,
            sigma=bunch_sigma,
            distribution=bunch_distribution,
            binomial_exponent=bunch_binomial_exponent
        )
        self._unit_charge_bunch._check_sampling(
            time_step=1 / (2 * max_frequency),
            half_window=self.bucket_length / 2
        )

        self._cached_profile: tuple[Real1dArray, Real1dArray] | None = None
        self._cached_spectrum: tuple[Real1dArray, Complex1dArray] | None = None

    @property
    def circumference(self) -> float:
        return self._circumference # m
    
    @property
    def max_frequency(self) -> float:
        return self._max_frequency # Hz

    @property
    def filling_scheme(self) -> Sequence[bool]:
        return self._filling_scheme
    
    @property
    def bunch_charge(self) -> float:
        return self._bunch_charge # C
    
    @property
    def bunch_sigma(self) -> float:
        return self._bunch_sigma # s
    
    @property
    def bunch_distribution(self) -> Literal['gaussian', 'binomial']:
        return self._bunch_distribution
    
    @property
    def bunch_binomial_exponent(self) -> float | None:
        return self._bunch_binomial_exponent
    
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
        return self.bunch_charge * self.num_bunches # Q
    
    @property
    def average_current(self) -> float:
        return self.charge * self.revolution_frequency # A

    def get_profile(self) -> tuple[Real1dArray, Real1dArray]:
        
        # if cached profile exists, return it instead of calculating again
        if self._cached_profile is not None:
            logger.debug(f'Using cached beam profile')
            return self._cached_profile # (s, A)

        # generate time axis centered for one bucket (t = 0 at bucket center)
        # with sampling such that the concatenated time-axis is equidistant
        delta_t = 1 / (2 * self.max_frequency)
        num_samples_per_bucket = ceil(self.bucket_length / delta_t)
        
        # time axis for one bunch
        bunch_time_array = np.linspace(
            start=-self.bucket_length/2, stop=self.bucket_length/2,
            num=num_samples_per_bucket, endpoint=False
        )

        # get profile of a bunch with unit charge
        unit_charge_bunch_profile = self._unit_charge_bunch.get_profile(bunch_time_array) # A

        empty_profile = np.zeros_like(bunch_time_array)

        # iterate over bunches to generate profile of entire beam
        bunch_profiles = []
        bunch_time_arrays = []

        logger.info(f'Generating beam profile: {self.harmonic_number} buckets ({self.num_bunches} filled), {num_samples_per_bucket} samples/bucket, {self.harmonic_number*num_samples_per_bucket} total samples')
        for bucket_index, is_filled in enumerate(
            tqdm(self.filling_scheme, disable=not logger.isEnabledFor(logging.INFO))
        ):
            bunch_time_arrays.append(bunch_time_array + bucket_index * self.bucket_length)
            if is_filled:
                bunch_profiles.append(self.bunch_charge * unit_charge_bunch_profile)
            else:
                bunch_profiles.append(empty_profile)
        beam_profile = np.concatenate(bunch_profiles) # A
        beam_time_array = np.concatenate(bunch_time_arrays)

        # store profile in cache
        self._cached_profile = (beam_time_array, beam_profile)

        return (
            beam_time_array, # s
            beam_profile # A
        )

    def get_spectrum(self) -> tuple[Real1dArray, Complex1dArray]:
        
        # if cached spectrum exists, return it instead of calculating again
        if self._cached_spectrum is not None:
            logger.debug(f'Using cached beam spectrum')
            return self._cached_spectrum # (Hz, C)
        
        # get time-domain profile of entire beam
        beam_time_array, beam_profile = self.get_profile() # A
        time_step = beam_time_array[1] - beam_time_array[0]

        logger.info(f'Computing beam spectrum: {len(beam_profile)} samples, frequency resolution {format_quantity(self.revolution_frequency, "Hz")}')

        # get beam spectrum
        spectrum_array = time_step * np.fft.rfft(beam_profile) # C
        frequency_array = np.fft.rfftfreq(len(beam_profile), time_step) # Hz

        # correct for non-zero start time, so that t = 0 is at the centre of the first bucket
        spectrum_array *= np.exp(-2j * pi * frequency_array * beam_time_array[0])

        # write generated spectrum to cache
        self._cached_spectrum = (frequency_array, spectrum_array)

        return (
            frequency_array, # Hz
            spectrum_array # C
        )
    
    def _get_real_impedance_at_beam_frequencies(
        self,
        frequency_array: Real1dArray, # Hz
        impedance_array: Complex1dArray, # Ohm
        negative_real_impedance_handling: NegativeImpedanceHandling
    ) -> Real1dArray:

        max_impedance_delta_f = float(max(np.diff(frequency_array)))
        if (max_impedance_delta_f > self.revolution_frequency):
            warnings.warn(
                f'Impedance frequency resolution {format_quantity(max_impedance_delta_f, "Hz")} is coarser than revolution frequency {format_quantity(self.revolution_frequency, "Hz")}, narrow-band features may be missed',
                AccuracyWarning, stacklevel=3
            )

        # impedance is assumed zero above its highest frequency
        max_impedance_frequency = float(frequency_array[-1])
        form_factor = self._unit_charge_bunch._get_form_factor_envelope(max_impedance_frequency)
        if form_factor > _MAX_FORM_FACTOR:
            warnings.warn(
                f'Impedance only given up to {format_quantity(max_impedance_frequency, "Hz")}, where the bunch spectrum is still {form_factor:.1e} of DC. Power loss at higher frequencies is neglected',
                AccuracyWarning, stacklevel=3
            )

        beam_freq, _ = self.get_spectrum() # Hz

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

        beam_freq, beam_spectrum = self.get_spectrum() # Hz, C
        real_impedance_at_beam_freq = self._get_real_impedance_at_beam_frequencies(
            frequency_array=frequency_array,
            impedance_array=impedance_array,
            negative_real_impedance_handling=negative_real_impedance_handling
        ) # Ohm

        power_loss_spectrum = (
            self.revolution_frequency**2
            * np.abs(beam_spectrum)**2 * real_impedance_at_beam_freq
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

        beam_freq, beam_spectrum = self.get_spectrum() # Hz, C
        real_impedance_at_beam_freq = self._get_real_impedance_at_beam_frequencies(
            frequency_array=frequency_array,
            impedance_array=value_array,
            negative_real_impedance_handling=negative_real_impedance_handling
        ) # Ohm

        frequency_step = beam_freq[1] - beam_freq[0]
        max_step = ceil(max_frequency_shift / frequency_step)
        shift_steps = np.arange(-max_step, max_step + 1)
        power_losses = np.zeros_like(shift_steps, dtype=float) # W

        logger.info(f'Calculating shifted power losses for frequency shifts +/- {format_quantity(max_frequency_shift, "Hz")} in steps of {format_quantity(frequency_step, "Hz")} ({len(shift_steps)} steps)')

        # precompute for speed
        squared_current = self.revolution_frequency**2 * np.abs(beam_spectrum)**2 # A^2

        for index, shift_step in enumerate(
            tqdm(shift_steps, disable=not logger.isEnabledFor(logging.INFO))
        ):
            shifted_impedance = np.roll(real_impedance_at_beam_freq, shift=shift_step)
            if shift_step < 0:
                shifted_impedance[shift_step:] = 0
            elif shift_step > 0:
                shifted_impedance[:shift_step] = 0
            
            # positive and negative frequencies contribute equally, except for DC
            power_losses[index] = (
                2 * float(np.sum(squared_current * shifted_impedance))
                - float(squared_current[0] * shifted_impedance[0])
            ) # A^2 * Ohm = W

        logger.info(f'Power loss calculation results:')
        logger.info(f'  max:    {format_quantity(max(power_losses), "W"):>12} at shift {format_quantity(shift_steps[np.argmax(power_losses)] * frequency_step, "Hz"):>12}')
        logger.info(f'  min:    {format_quantity(min(power_losses), "W"):>12} at shift {format_quantity(shift_steps[np.argmin(power_losses)] * frequency_step, "Hz"):>12}')
        logger.info(f'  mean:   {format_quantity(float(np.mean(power_losses)), "W"):>12}')
        logger.info(f'  median: {format_quantity(float(np.median(power_losses)), "W"):>12}')
        
        return (
            frequency_step * shift_steps, # Hz
            power_losses # W
        )
