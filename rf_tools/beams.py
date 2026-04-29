import numpy as np
from scipy.constants import c, e
from math import pi, gamma, sqrt, ceil
from tqdm import tqdm

from typing import Literal, Sequence
from .quantities import RealArray, ComplexArray


def get_binomial_bunch_profile(
    time_array: RealArray, # s
    length_4sigma: float, # s
    exponent: float,
) -> tuple[RealArray, RealArray]:

    full_bunch_length = length_4sigma * sqrt(3 + 2 * exponent) / 2
    amplitude = (
        2 * gamma(1.5 + exponent)
        / (full_bunch_length * sqrt(pi) * gamma(1 + exponent))
    )
    profile = (1 - 4 * (time_array / full_bunch_length) ** 2) ** exponent
    profile[np.abs(time_array) > full_bunch_length / 2] = 0
    
    return time_array, amplitude * profile # 1/s


def get_gaussian_bunch_profile(
    time_array: RealArray, # s
    length_4sigma: float # s
) -> tuple[RealArray, RealArray]:

    sigma = length_4sigma / 4
    amplitude = 1 / (sigma * sqrt(2 * pi))
    profile = np.exp(-0.5 * (time_array / sigma) ** 2)
    
    return time_array, amplitude * profile # 1/s


class Bunch:

    def __init__(
        self,
        distribution: Literal['binomial', 'gaussian'],
        length_4sigma: float, # s
        intensity: float, # particles per bunch
        charge_number: int = 1,
        binomial_exponent: float | None = None
    ) -> None:
        
        self.distribution = distribution
        self.length_4sigma = length_4sigma
        self.intensity = intensity
        self.charge_number = charge_number

        self.binomial_exponent = binomial_exponent

    @property
    def length_sigma(self) -> float:
        return self.length_sigma / 4 # s

    @property
    def charge(self) -> float:
        return self.charge_number * e * self.intensity # C
    
    def get_profile(
        self,
        time_array: RealArray # s
    ) -> tuple[RealArray, RealArray]:

        if self.distribution == 'binomial':
            if not self.binomial_exponent:
                raise ValueError('Must specify binomial exponent (mu) for binomial bunch distribution')
            _, profile = get_binomial_bunch_profile(
                time_array=time_array,
                length_4sigma=self.length_4sigma,
                exponent=self.binomial_exponent
            )

        elif self.distribution == 'gaussian':
            if self.binomial_exponent is not None:
                raise ValueError('Binomial exponent (mu) should not be specified for gaussian bunch distribution')
            _,profile = get_gaussian_bunch_profile(
                time_array=time_array,
                length_4sigma=self.length_4sigma
            )

        else:
            raise ValueError(f'Unknown bunch distribution `{self.distribution}`')
        
        assert np.isclose(np.trapezoid(profile, time_array), 1), f'Integral over `{self.distribution}` normalized bunch profile not equal to 1'
        
        return (
            time_array, # s
            self.charge * profile # A
        )

    def get_spectrum(
        self,
        max_frequency: float, # Hz
        frequency_step: float # Hz
    ) -> tuple[RealArray, ComplexArray]:
        
        time_period = 1 / frequency_step
        time_step = 1 / (2 * max_frequency)
        num_samples = ceil(time_period / time_step)
        time_array = np.linspace(
            start=-time_period/2, stop=time_period/2, num=num_samples, endpoint=False
        )

        _, profile = self.get_profile(time_array)
        spectrum_array = np.fft.rfft(profile) * time_step
        frequency_array = np.fft.rfftfreq(len(profile), time_step)

        # Sanity checks
        assert np.isclose(spectrum_array[0], self.charge), f'DC component of spectrum ({spectrum_array[0]}) not equal to bunch charge ({self.charge})'
        
        return (
            frequency_array, # Hz
            spectrum_array # C
        )


class Beam:

    def __init__(
        self,
        bunch: Bunch,
        circumference: float, # m
        max_frequency: float, # Hz
        filling_scheme: Sequence[bool],
        beta: float = 1.0
    ) -> None:
        
        self._bunch = bunch
        self._circumference = circumference
        self._max_frequency = max_frequency

        if not any(filling_scheme):
            raise ValueError('Filling scheme must contain at least one filled bucket (`True`)')
        self._filling_scheme = filling_scheme

        if beta != 1.0:
            raise NotImplementedError('Only relativistic beams with `beta` = 1 are currently supported')

        self._cached_profile: tuple[RealArray, RealArray] | None = None
        self._cached_spectrum: tuple[RealArray, ComplexArray] | None = None

    @property
    def bunch(self) -> Bunch:
        return self._bunch
    
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
        return self.bunch.charge * self.num_bunches # Q
    
    @property
    def average_current(self) -> float:
        return self.charge * self.revolution_frequency # A

    def get_profile(
        self,
        silent: bool = False
    ) -> tuple[RealArray, RealArray]:
        
        if self._cached_profile is not None:
            if not silent:
                print(f'Using cached beam profile')
            return self._cached_profile

        # generate time axis centered for one bucket (t = 0 at bucket center)
        # with sampling such that the concatenated time-axis is equidistant
        delta_t = 1 / (2 * self.max_frequency)
        num_samples_per_bucket = ceil(self.bucket_length / delta_t)
        
        bucket_time_array = np.linspace(
            start=-self.bucket_length/2, stop=self.bucket_length/2,
            num=num_samples_per_bucket, endpoint=False
        )

        bunch_time_array, bunch_profile = self.bunch.get_profile(bucket_time_array)
        empty_profile = np.zeros_like(bucket_time_array)

        bunch_profiles = []
        bunch_time_arrays = []
        if not silent:
            print(f'Generating beam profile: {self.harmonic_number} buckets ({self.num_bunches} filled), {num_samples_per_bucket} samples/bucket, {self.harmonic_number*num_samples_per_bucket} total samples')
        for n, is_filled in enumerate(tqdm(self.filling_scheme, disable=silent)):
            bunch_time_arrays.append(bunch_time_array + n * self.bucket_length)
            if is_filled:
                bunch_profiles.append(bunch_profile)
            else:
                bunch_profiles.append(empty_profile)

        beam_profile = np.concatenate(bunch_profiles)
        beam_time_array = np.concatenate(bunch_time_arrays)

        # Sanity checks
        beam_time_array_check = np.linspace(
            start=-self.bucket_length/2,
            stop=self.revolution_period - self.bucket_length/2,
            num=self.harmonic_number * num_samples_per_bucket, endpoint=False
        )
        assert len(beam_time_array) == len(beam_time_array_check), f'Number of samples in time array ({len(beam_time_array)}) and beam profile ({len(beam_profile)}) do not match'
        assert np.allclose(beam_time_array, beam_time_array_check), f'Samples in time array do not match expected values'
        integrated_beam_current = np.trapezoid(beam_profile, beam_time_array)
        assert np.isclose(integrated_beam_current, self.charge), f'Integral over beam profile ({integrated_beam_current} C) not equal to total beam charge ({self.charge} C)'

        self._cached_profile = (beam_time_array, beam_profile)

        return (
            beam_time_array, # s
            beam_profile # A
        )

    def get_spectrum(
        self,
        silent: bool = False
    ) -> tuple[RealArray, ComplexArray]:
        
        if self._cached_spectrum is not None:
            if not silent:
                print(f'Using cached beam spectrum')
            return self._cached_spectrum
        
        beam_time_array, beam_profile = self.get_profile(
            silent=silent
        )
        time_step = beam_time_array[1] - beam_time_array[0]

        if not silent:
            print(f'Computing beam spectrum: {len(beam_profile)} samples, frequency resolution {self.revolution_frequency} Hz')
        
        spectrum_array = self.revolution_frequency * time_step * np.fft.rfft(beam_profile) 
        frequency_array = np.fft.rfftfreq(len(beam_profile), time_step)

        # Sanity checks
        assert np.isclose(spectrum_array[0], self.average_current), f'DC component of spectrum ({spectrum_array[0]} A) not equal average beam current ({self.average_current} A)'
        assert np.isclose(frequency_array[1]-frequency_array[0], self.revolution_frequency), f'Frequency resolution of spectrum ({frequency_array[1]-frequency_array[0]} Hz) does not match revolution frequency ({self.revolution_frequency} Hz)'

        self._cached_spectrum = (frequency_array, spectrum_array)

        return (
            frequency_array, # Hz
            spectrum_array # A
        )
    
    def get_power_loss_spectrum(
        self,
        impedance_frequency_array: RealArray, # Hz
        impedance_value_array: ComplexArray, # Ohm
        silent: bool = False
    ) -> tuple[RealArray, RealArray]:

        max_impedance_delta_f = max(np.diff(impedance_frequency_array))
        if (max_impedance_delta_f > self.revolution_frequency) and not silent:
            print(f'WARNING: Impedance frequency resolution ({max_impedance_delta_f} Hz) is coarser than revolution frequency ({self.revolution_frequency} Hz). This may lead to inaccurate results')
        
        beam_freq, beam_spectrum = self.get_spectrum(silent=silent)

        if (beam_freq[0] > impedance_frequency_array[0]) or (beam_freq[-1] < impedance_frequency_array[-1]):
            print(f'WARNING: Beam spectrum frequency range ({beam_freq[0]} - {beam_freq[-1]} Hz) does not cover impedance frequency range ({impedance_frequency_array[0]} - {impedance_frequency_array[-1]} Hz). This may lead to inaccurate results')

        impedance_values_at_beam_freq = np.interp(
            x=beam_freq,
            xp=impedance_frequency_array, fp=impedance_value_array,
            left=0.0, right=0.0
        )

        power_loss_spectrum = 2 * np.abs(beam_spectrum)**2 * np.real(impedance_values_at_beam_freq)
    
        # Sanity checks
        num_negative = sum(power_loss_spectrum < 0)
        if (num_negative > 0) and not silent:
            print(f'WARNING: Power loss spectrum contains {num_negative} ({100 * num_negative / len(power_loss_spectrum):.1f} % of samples) negative values, likely because of negative real part of the impeadance.')

        return (
            beam_freq, # Hz
            power_loss_spectrum # W
        )

    def get_power_loss(
        self,
        impedance_frequency_array: RealArray, # Hz
        impedance_value_array: ComplexArray, # Ohm
        silent: bool = False
    ) -> float:

        _, power_loss_spectrum = self.get_power_loss_spectrum(
            impedance_frequency_array=impedance_frequency_array,
            impedance_value_array=impedance_value_array,
            silent=silent
        )
        power_loss = float(np.sum(power_loss_spectrum))

        # Sanity checks
        assert power_loss >= 0, f'Power loss should be non-negative, got {power_loss} W'

        return power_loss # W
    
    def get_shifted_power_loss(
        self,
        impedance_frequency_array: RealArray, # Hz
        impedance_value_array: ComplexArray, # Ohm
        frequency_shift: float, # Hz
        silent: bool = False
    ) -> tuple[RealArray, RealArray]:
        
        if frequency_shift is None:
            frequency_shift = self.revolution_frequency
            if not silent:
                print(f'No `frequency_shift_range` provided, shifting by revolution frequency: +/- {frequency_shift} Hz')

        # calculate sufficient turns to sample the impedance
        beam_freq, beam_spectrum = self.get_spectrum(silent=silent)

        frequency_step = beam_freq[1] - beam_freq[0]
        frequency_shifts = np.arange(
            -frequency_shift,
            frequency_shift + frequency_step,
            frequency_step
        )
        power_losses = np.zeros_like(frequency_shifts)

        if not silent:
            print(f'Calculating shifted power losses for frequency shifts +/- {frequency_shift} Hz in steps of {frequency_step} Hz ({len(frequency_shifts)} steps)')

        for n, shift in enumerate(tqdm(frequency_shifts, disable=silent)):
            shifted_impedance = np.interp(
                x=beam_freq,
                xp=impedance_frequency_array - shift,
                fp=impedance_value_array
            )
            power_losses[n] = self.get_power_loss(
                impedance_frequency_array=beam_freq,
                impedance_value_array=shifted_impedance,
                silent=True
            )
        
        return (
            frequency_shifts, # Hz
            power_losses # W
        )