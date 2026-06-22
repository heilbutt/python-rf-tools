import numpy as np
from scipy.constants import c
from math import pi, gamma, sqrt, ceil
from tqdm import tqdm

from .units import format_quantity

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
        charge: float, # C
        length_4sigma: float, # s
        distribution: str,
        binomial_exponent: float | None = None,
        sanity_check_rtol: float = 1e-5,
    ) -> None:
        
        self.distribution = distribution
        self.length_4sigma = length_4sigma
        self.charge = charge
        self.binomial_exponent = binomial_exponent

        self.sanity_check_rtol = sanity_check_rtol

    @property
    def length_sigma(self) -> float:
        return self.length_4sigma / 4 # s
    
    def get_profile(
        self,
        time_array: RealArray # s
    ) -> tuple[RealArray, RealArray]:

        # get normalized bunch profile according to distribution, unit is 1/s
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
            _, profile = get_gaussian_bunch_profile(
                time_array=time_array,
                length_4sigma=self.length_4sigma
            )
        else:
            raise ValueError(f'Unknown bunch distribution `{self.distribution}`')
        
        # Sanity checks
        integral = np.trapezoid(profile, time_array)
        assert np.isclose(integral, 1, rtol=self.sanity_check_rtol), f'Integral over `{self.distribution}` normalized bunch profile not equal to unity. Got {integral} instead'
        
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
        assert np.isclose(spectrum_array[0], self.charge, rtol=self.sanity_check_rtol), f'DC component of spectrum {format_quantity(spectrum_array[0], "C")}) not equal to bunch charge {format_quantity(self.charge, "C")}'
        
        return (
            frequency_array, # Hz
            spectrum_array # C
        )


class Beam:

    def __init__(
        self,
        circumference: float, # m
        max_frequency: float, # Hz
        filling_scheme: Sequence[bool],
        bunch_charge: float, # C
        bunch_length_4sigma: float, # s
        bunch_distribution: str,
        bunch_binomial_exponent: float | None = None,
        beta: float = 1.0,
        sanity_check_rtol: float = 1e-5,
        silent: bool = False,
    ) -> None:
        
        self._circumference = circumference
        self._max_frequency = max_frequency
        if not any(filling_scheme):
            raise ValueError('Filling scheme must contain at least one filled bucket (`True`)')
        self._filling_scheme = filling_scheme
        self._bunch_charge = bunch_charge
        self._bunch_length_4sigma = bunch_length_4sigma
        self._bunch_distribution = bunch_distribution
        self._bunch_binomial_exponent = bunch_binomial_exponent
        if beta != 1.0:
            raise NotImplementedError('Only relativistic beams with `beta` = 1 are currently supported')
        
        self.silent = silent
        self.sanity_check_rtol = sanity_check_rtol

        self._cached_profile: tuple[RealArray, RealArray] | None = None
        self._cached_spectrum: tuple[RealArray, ComplexArray] | None = None

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
    def bunch_length_4sigma(self) -> float:
        return self._bunch_length_4sigma # s
    
    @property
    def bunch_length_sigma(self) -> float:
        return self.bunch_length_4sigma / 4 # s
    
    @property
    def bunch_distribution(self) -> str:
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

    def get_profile(self) -> tuple[RealArray, RealArray]:
        
        # if cached profile exists, return it instead of calculating again
        if self._cached_profile is not None:
            if not self.silent:
                print(f'Using cached beam profile')
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
        _, unit_charge_bunch_profile = Bunch(
            charge=1,
            length_4sigma=self.bunch_length_4sigma,
            distribution=self.bunch_distribution,
            binomial_exponent=self.bunch_binomial_exponent,
            sanity_check_rtol=self.sanity_check_rtol
        ).get_profile(bunch_time_array) # A

        empty_profile = np.zeros_like(bunch_time_array)

        # iterate over bunches to generate profile of entire beam
        bunch_profiles = []
        bunch_time_arrays = []
        if not self.silent:
            print(f'Generating beam profile: {self.harmonic_number} buckets ({self.num_bunches} filled), {num_samples_per_bucket} samples/bucket, {self.harmonic_number*num_samples_per_bucket} total samples')
        for bucket_index, is_filled in enumerate(tqdm(self.filling_scheme, disable=self.silent)):
            bunch_time_arrays.append(bunch_time_array + bucket_index * self.bucket_length)
            if is_filled:
                bunch_profiles.append(self.bunch_charge * unit_charge_bunch_profile)
            else:
                bunch_profiles.append(empty_profile)
        beam_profile = np.concatenate(bunch_profiles) # A
        beam_time_array = np.concatenate(bunch_time_arrays)

        # Sanity checks
        # time array of the entire beam 
        beam_time_array_check = np.linspace(
            start=-self.bucket_length/2,
            stop=self.revolution_period - self.bucket_length/2,
            num=self.harmonic_number * num_samples_per_bucket, endpoint=False
        )
        assert len(beam_time_array) == len(beam_time_array_check), f'Number of samples in time array ({len(beam_time_array)}) and beam profile ({len(beam_profile)}) do not match'
        assert np.allclose(beam_time_array, beam_time_array_check, rtol=self.sanity_check_rtol), f'Samples in time array do not match expected values'
        integrated_beam_current = float(np.trapezoid(beam_profile, beam_time_array))
        assert np.isclose(integrated_beam_current, self.charge, rtol=self.sanity_check_rtol), f'Integral over beam profile {format_quantity(integrated_beam_current, "C")}) not equal to total beam charge {format_quantity(self.charge, "C")}'

        # store profile in cache
        self._cached_profile = (beam_time_array, beam_profile)

        return (
            beam_time_array, # s
            beam_profile # A
        )

    def get_spectrum(self) -> tuple[RealArray, ComplexArray]:
        
        # if cached spectrum exists, return it instead of calculating again
        if self._cached_spectrum is not None:
            if not self.silent:
                print(f'Using cached beam spectrum')
            return self._cached_spectrum # (Hz, C)
        
        # get time-domain profile of entire beam
        beam_time_array, beam_profile = self.get_profile() # A
        time_step = beam_time_array[1] - beam_time_array[0]

        if not self.silent:
            print(f'Computing beam spectrum: {len(beam_profile)} samples, frequency resolution {format_quantity(self.revolution_frequency, "Hz")}')

        # get beam spectrum
        spectrum_array = time_step * np.fft.rfft(beam_profile) # C
        frequency_array = np.fft.rfftfreq(len(beam_profile), time_step) # Hz

        # Sanity checks
        assert np.isclose(spectrum_array[0], self.charge, rtol=self.sanity_check_rtol), f'DC component of spectrum {format_quantity(spectrum_array[0], "C")}) not equal average beam charge {format_quantity(self.charge, "C")}'
        assert np.isclose(frequency_array[1]-frequency_array[0], self.revolution_frequency, rtol=self.sanity_check_rtol), f'Frequency resolution of spectrum {format_quantity(frequency_array[1]-frequency_array[0], "Hz")}) does not match revolution frequency {format_quantity(self.revolution_frequency, "Hz")}'

        # write generated spectrum to cache
        self._cached_spectrum = (frequency_array, spectrum_array)

        return (
            frequency_array, # Hz
            spectrum_array # C
        )
    
    def get_power_loss_spectrum(
        self,
        impedance_frequency_array: RealArray, # Hz
        impedance_value_array: ComplexArray, # Ohm
        negative_real_impedance_handling: Literal['ignore', 'clip', 'abs'] = 'ignore'
    ) -> tuple[RealArray, RealArray]:

        max_impedance_delta_f = max(np.diff(impedance_frequency_array))
        if (max_impedance_delta_f > self.revolution_frequency) and not self.silent:
            print(f'WARNING: Impedance frequency resolution {format_quantity(max_impedance_delta_f, "Hz")} is coarser than revolution frequency {format_quantity(self.revolution_frequency, "Hz")}. This may lead to inaccurate results')
        
        beam_freq, beam_spectrum = self.get_spectrum() # Hz, C

        if (
            (beam_freq[0] > impedance_frequency_array[0])
            or (beam_freq[-1] < impedance_frequency_array[-1])
        ) and not self.silent:
            print(f'WARNING: Beam spectrum frequency range {format_quantity(beam_freq[0], "Hz")} - {format_quantity(beam_freq[-1], "Hz")} does not cover impedance frequency range {format_quantity(impedance_frequency_array[0], "Hz")} - {format_quantity(impedance_frequency_array[-1], "Hz")}. This may lead to inaccurate results')

        real_impedance_at_beam_freq = np.interp(
                x=beam_freq,
                xp=impedance_frequency_array,
                fp=np.real(impedance_value_array),
                left=0.0, right=0.0
        )
        
        if negative_real_impedance_handling == 'clip':
            real_impedance_at_beam_freq = np.clip(real_impedance_at_beam_freq, min=0, max=None)
        elif negative_real_impedance_handling == 'abs':
            real_impedance_at_beam_freq = np.abs(real_impedance_at_beam_freq)
        elif negative_real_impedance_handling == 'ignore':
            pass
        else:
            raise ValueError(f'Unknown handling of negative impedance real part: `{negative_real_impedance_handling}`')

        power_loss_spectrum = (
            2 * self.revolution_frequency**2
            * np.abs(beam_spectrum)**2 * real_impedance_at_beam_freq
        ) # Hz^2 * C^2 * Ohm = W
    
        # Sanity checks
        num_negative = sum(power_loss_spectrum < 0)
        if (num_negative > 0) and not self.silent:
            print(f'WARNING: Power loss spectrum contains {num_negative} ({100 * num_negative / len(power_loss_spectrum):.1f} % of samples) negative values, likely because of negative real part of the impedance.')

        return (
            beam_freq, # Hz
            power_loss_spectrum # W
        )

    def get_power_loss(
        self,
        impedance_frequency_array: RealArray, # Hz
        impedance_value_array: ComplexArray, # Ohm
        negative_real_impedance_handling: Literal['ignore', 'clip', 'abs'] = 'ignore',
    ) -> float:

        _, power_loss_spectrum = self.get_power_loss_spectrum(
            impedance_frequency_array=impedance_frequency_array,
            impedance_value_array=impedance_value_array,
            negative_real_impedance_handling=negative_real_impedance_handling
        ) # W
        power_loss = float(np.sum(power_loss_spectrum)) # W

        # Sanity checks
        assert power_loss >= 0, f'Power loss should be non-negative, got {power_loss} W'

        return power_loss # W
    
    def get_shifted_power_loss(
        self,
        impedance_frequency_array: RealArray, # Hz
        impedance_value_array: ComplexArray, # Ohm
        max_frequency_shift: float, # Hz
        negative_real_impedance_handling: Literal['ignore', 'clip', 'abs'] = 'ignore'
    ) -> tuple[RealArray, RealArray]:

        beam_freq, beam_spectrum = self.get_spectrum() # Hz, C

        if (
            (beam_freq[0] > impedance_frequency_array[0])
            or (beam_freq[-1] < impedance_frequency_array[-1])
        ) and not self.silent:
            print(f'WARNING: Beam spectrum frequency range {format_quantity(beam_freq[0], "Hz")} - {format_quantity(beam_freq[-1], "Hz")} does not cover impedance frequency range {format_quantity(impedance_frequency_array[0], "Hz")} - {format_quantity(impedance_frequency_array[-1], "Hz")}. This may lead to inaccurate results')

        real_impedance_at_beam_freq = np.interp(
                x=beam_freq,
                xp=impedance_frequency_array,
                fp=np.real(impedance_value_array),
                left=0.0, right=0.0
        )
        
        if negative_real_impedance_handling == 'clip':
            real_impedance_at_beam_freq = np.clip(real_impedance_at_beam_freq, min=0, max=None)
        elif negative_real_impedance_handling == 'abs':
            real_impedance_at_beam_freq = np.abs(real_impedance_at_beam_freq)
        elif negative_real_impedance_handling == 'ignore':
            pass
        else:
            raise ValueError(f'Unknown handling of negative impedance real part: `{negative_real_impedance_handling}`')

        frequency_step = beam_freq[1] - beam_freq[0]
        max_step = ceil(max_frequency_shift / frequency_step)
        shift_steps = np.arange(-max_step, max_step + 1)
        power_losses = np.zeros_like(shift_steps, dtype=float) # W

        if not self.silent:
            print(f'Calculating shifted power losses for frequency shifts +/- {format_quantity(max_frequency_shift, "Hz")} in steps of {format_quantity(frequency_step, "Hz")} ({len(shift_steps)} steps)')

        # precompute for speed
        squared_current = 2 * self.revolution_frequency**2 * np.abs(beam_spectrum)**2 # A^2

        for index, shift_step in enumerate(tqdm(shift_steps, disable=self.silent)):
            shifted_impedance = np.roll(real_impedance_at_beam_freq, shift=shift_step)
            if shift_step < 0:
                shifted_impedance[shift_step:] = 0
            elif shift_step > 0:
                shifted_impedance[:shift_step] = 0
            power_losses[index] = float(
                np.sum(squared_current * shifted_impedance) # A^2 * Ohm = W
            )

        if not self.silent:
            print(f'Power loss calculation results:')
            print(f'  max:    {format_quantity(max(power_losses), "W"):>12} at shift {format_quantity(shift_steps[np.argmax(power_losses)] * frequency_step, "Hz"):>12}')
            print(f'  min:    {format_quantity(min(power_losses), "W"):>12} at shift {format_quantity(shift_steps[np.argmin(power_losses)] * frequency_step, "Hz"):>12}')
            print(f'  mean:   {format_quantity(float(np.mean(power_losses)), "W"):>12}')
            print(f'  median: {format_quantity(float(np.median(power_losses)), "W"):>12}')
        
        return (
            frequency_step * shift_steps, # Hz
            power_losses # W
        )
