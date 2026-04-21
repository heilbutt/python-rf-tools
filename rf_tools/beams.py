import numpy as np
from scipy.constants import c, e
from math import pi, gamma, sqrt
from tqdm import tqdm

from typing import Literal, Sequence, Any
from numpy.typing import NDArray

from .quantities import RealQuantity, ComplexQuantity


def _get_binomial_bunch_profile(
    time_array: NDArray[np.floating[Any]],
    length_4sigma: float,
    exponent: float,
) -> NDArray[np.floating[Any]]:

    full_bunch_length = length_4sigma * sqrt(3 + 2 * exponent) / 2
    amplitude = (
        2 * gamma(1.5 + exponent)
        / (full_bunch_length * sqrt(pi) * gamma(1 + exponent))
    )
    profile = amplitude * (1 - 4 * (time_array / full_bunch_length) ** 2) ** exponent
    profile[np.abs(time_array) > full_bunch_length / 2] = 0
    
    return profile


class Bunch:

    def __init__(
        self,
        distribution: Literal['binomial'],
        length_4sigma: float,
        intensity: float,
        charge_number: int = 1,
        binomial_exponent: float | None = None
    ) -> None:
        
        self.distribution = distribution
        self.length_4sigma = length_4sigma
        self.intensity = intensity
        self.charge_number = charge_number

        self.binomial_exponent = binomial_exponent

    @property
    def length_1sigma(self) -> float:
        return self.length_4sigma / 4

    @property
    def charge(self) -> float:
        return self.charge_number * e * self.intensity
    
    def get_profile(
        self,
        time_array: NDArray[np.floating[Any]]
    ) -> RealQuantity:

        if self.distribution == 'binomial':
            if not self.binomial_exponent:
                raise ValueError('Must specify binomial exponent (mu) for binomial bunch distribution')
            profile = _get_binomial_bunch_profile(
                time_array=time_array,
                length_4sigma=self.length_4sigma,
                exponent=self.binomial_exponent
            )

        else:
            raise ValueError(f'Unknown bunch distribution `{self.distribution}`')
        
        return RealQuantity(time_array, self.charge * profile)
            

class Beam:

    def __init__(
        self,
        bunch: Bunch,
        bucket_length: float,
        filling_scheme: Sequence[bool]
    ) -> None:
        
        self.bunch = bunch
        self.bucket_length = bucket_length
        self.filling_scheme = filling_scheme

    def get_profile(
        self,
        max_frequency: float,
        disable_progress: bool = False,
    ) -> RealQuantity:

        # generate time axis centered for one bucket (t = 0 at bucket center)
        # with sampling such that the concatenated time-axis is equidistant
        num_buckets = len(self.filling_scheme)
        delta_t = 1 / (2 * max_frequency)
        num_samples_per_bucket = int(self.bucket_length * delta_t + 0.5)
        bucket_time_array = np.linspace(
            start=-self.bucket_length/2, stop=self.bucket_length/2,
            num=num_samples_per_bucket, endpoint=False
        )
        beam_time_array = np.linspace(
            start=-self.bucket_length/2,
            stop=(num_buckets - 0.5) * self.bucket_length,
            num=num_buckets * num_samples_per_bucket, endpoint=False
        )

        bunch_profile = self.bunch.get_profile(bucket_time_array)
        empty_profile = np.zeros_like(bucket_time_array)
        bunch_profiles = []
        for is_filled in tqdm(
            self.filling_scheme,
            desc='Generating beam profile: ',
            disable=disable_progress
        ):
            if is_filled:
                bunch_profiles.append(bunch_profile)
            else:
                bunch_profiles.append(empty_profile)

        return RealQuantity(beam_time_array, np.concatenate(bunch_profiles))
    
    def get_spectrum(
        self,
        max_frequency: float,
        delta_frequency: float | None
    ) -> ComplexQuantity:
        
        beam_profile = self.get_profile(max_frequency=max_frequency)
        time_period = beam_profile.x[-1] - beam_profile.x[0]
        time_step = beam_profile.x[1] - beam_profile.x[0]

        if delta_frequency is not None:
            num_turns = int(1 / (time_period * delta_frequency) + 0.5)
        else:
            num_turns = 1
        profile_array = np.tile(beam_profile.y, num_turns)

        spectrum_array = np.fft.rfft(profile_array) * time_step
        frequency_array = np.fft.rfftfreq(len(spectrum_array), time_step)

        return ComplexQuantity(frequency_array, spectrum_array)