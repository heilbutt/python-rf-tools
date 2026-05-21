from math import pi, sqrt
import numpy as np

from .quantities import RealArray, ComplexArray


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
        return self.omega_r * sqrt(1 - 0.25 / self.reduced_quality_factor**2) # rad/s


class LongitudinalResonator(_Resonator):
    
    def get_loss_factor(
        self,
        bunch_length_sigma: float = 0 # s
    ) -> float:
        
        # Zotter 5.1.1
        if self.omega_r * bunch_length_sigma > 0.1:
            print(f'WARNING: Loss factor calculation valid for short bunches with omega_r * bunch_sigma << 1, but got {self.omega_r * bunch_length_sigma:.2f}')

        return self.omega_r / 2 * self.R_over_Q * (
            1 - 2 / pi * self.omega_r * bunch_length_sigma / self.quality_factor
        ) # V/C
        
    def get_impedance(
        self,
        frequency_array: RealArray,
        wake_length: float | None = None # s
    ) -> tuple[RealArray, ComplexArray]:

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

        return (
            frequency_array, # Hz
            impedance # Ohm (or whatever unit of shunt impedance was)
        )
    
    def get_wake_function(
        self,
        time_array: RealArray,
        zero_time_tolerance: float = 1e-12 # s
    ) -> tuple[RealArray, RealArray]:
        
        wake_function = (
            self.omega_r * self.R_over_Q
            * np.exp(-self.damping_coefficient * time_array)
            * (
                np.cos(self.reduced_omega_r * time_array)
                - np.sin(self.reduced_omega_r * time_array) / 2 / self.reduced_quality_factor
            )
        )

        # causal wake function
        wake_function[time_array < -zero_time_tolerance] = 0 
        # fundamental beam-loading theorem
        wake_function[np.abs(time_array) < zero_time_tolerance] /= 2

        return (
            time_array, # s
            wake_function # V/C (or whatever unit of shunt impedance was)
        )
    

class TransverseResonator(_Resonator):
    
    def get_impedance(
        self,
        frequency_array: RealArray,
        wake_length: float | None = None # s
    ) -> tuple[RealArray, ComplexArray]:

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

        return (
            frequency_array, # Hz
            impedance # Ohm (or whatever unit of shunt impedance was)
        )

    def get_wake_function(
        self,
        time_array: RealArray,
        zero_time_tolerance: float = 1e-12 # s
    ) -> tuple[RealArray, RealArray]:

        wake_function = (
            self.omega_r * self.R_over_Q
            * np.exp(-self.damping_coefficient * time_array)
            * np.sin(self.reduced_omega_r * time_array)
        )

        # causal wake function
        wake_function[time_array < -zero_time_tolerance] = 0 

        return (
            time_array, # s
            wake_function # V/C/m (or whatever unit of shunt impedance was)
        )