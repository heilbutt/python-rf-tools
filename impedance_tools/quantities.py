import numpy as np

from typing import Iterable, Any
from numpy.typing import NDArray


class FrequencyMismatchError(Exception):
    pass


class UnequalSampleCountError(Exception):
    pass


class Spectrum:

    def __init__(self, f: Iterable[float], z: Iterable[complex]) -> None:

        self._f = np.asarray(f, dtype=float)
        self._z = np.asarray(z, dtype=complex)

        if len(self._f) != len(self._z):
            raise UnequalSampleCountError(f'Frequency and value have unequal number of samples! (f: `{len(self._f)}`, y: `{len(self._z)}`)')

    @property
    def n(self) -> int:
        return len(self._f)
    
    @property
    def f(self) -> NDArray[np.floating[Any]]:
        return self._f

    @f.setter
    def f(self, new_values: Iterable[float]) -> None:
        new_values = np.asarray(new_values, dtype=float)
        if not self.n == len(new_values):
            raise UnequalSampleCountError(f'Changing number of frequency samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        self._f = new_values

    @property
    def z(self) -> NDArray[np.complexfloating[Any]]:
        return self._z

    @z.setter
    def z(self, new_values: Iterable[complex]) -> None:
        new_values = np.asarray(new_values, dtype=complex)
        if not self.n == len(new_values):
            raise UnequalSampleCountError(f'Changing number of spectrum samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        self._z = new_values

    @property
    def real(self) -> NDArray[np.floating[Any]]:
        return np.real(self._z)
    
    @real.setter
    def real(self, new_values: Iterable[float]) -> None:
        new_values = np.asarray(new_values, dtype=float)
        if not self.n == len(new_values):
            raise UnequalSampleCountError(f'Changing number of spectrum samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        self._z = new_values + 1j*np.imag(self._z)
    
    @property
    def imag(self) -> NDArray[np.floating[Any]]:
        return np.imag(self._z)
    
    @imag.setter
    def imag(self, new_values: Iterable[float]) -> None:
        new_values = np.asarray(new_values, dtype=float)
        if not self.n == len(new_values):
            raise UnequalSampleCountError(f'Changing number of spectrum samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        self._z = np.real(self._z) + 1j*new_values
    
    @property
    def mag(self) -> NDArray[np.floating[Any]]:
        return np.abs(self._z)
    
    @mag.setter
    def mag(self, new_values: Iterable[float]) -> None:
        new_values = np.asarray(new_values, dtype=float)
        if not self.n == len(new_values):
            raise UnequalSampleCountError(f'Changing number of spectrum samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        self._z = np.abs(new_values) * np.exp(1j*np.angle(self._z))
    
    @property
    def phase(self) -> NDArray[np.floating[Any]]:
        return np.angle(self._z)
    
    @phase.setter
    def phase(self, new_values: Iterable[float]) -> None:
        new_values = np.asarray(new_values, dtype=float)
        if not self.n == len(new_values):
            raise UnequalSampleCountError(f'Changing number of spectrum samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        self._z = np.abs(self._z) * np.exp(1j*new_values)

    def __neg__(self) -> Spectrum:
        return type(self)(self._f, -self._z)

    def __add__(self, other: Spectrum | complex) -> Spectrum :
        if isinstance(other, Spectrum):
            if not np.allclose(self._f, other._f):
                raise FrequencyMismatchError('Frequencies do not match for spectrum addition')
            return type(self)(self._f, self._z + other._z)
        if isinstance(other, (int, float, complex)):
            return type(self)(self._f, self._z + other)
        return NotImplemented

    def __radd__(self, other: Spectrum | complex) -> Spectrum:
        return self + other

    def __sub__(self, other: Spectrum | complex) -> Spectrum:
        if isinstance(other, Spectrum):
            if not np.allclose(self._f, other._f):
                raise FrequencyMismatchError('Frequencies do not match for spectrum subtraction')
            return type(self)(self._f, self._z - other._z)
        if isinstance(other, (int, float, complex)):
            return type(self)(self._f, self._z - other)
        return NotImplemented
    
    def __rsub__(self, other: Spectrum | complex) -> Spectrum:
        return -self + other

    def __mul__(self, other: complex) -> Spectrum:
        if isinstance(other, (int, float, complex)):
            return type(self)(self._f, other * self._z)
        return NotImplemented

    def __rmul__(self, other: complex) -> Spectrum:
        return self * other
    
    def __truediv__(self, other: complex) -> Spectrum:
        if isinstance(other, (int, float, complex)):
            return type(self)(self._f, self._z / other)
        return NotImplemented


#     @classmethod
#     def from_wake(
#         cls,
#         wake: Wake,
#         delta_f: float | None = None,
#         bunch_sigma: float | None = None,
#         cutoff_at_n_sigma: float | None = 2
#         ) -> Impedance:

#         ds = wake.s[1] - wake.s[0] # m, distance step

#         if delta_f is None:
#             # compute spectrum using the natural frequencies from wake potential
#             n_samples = len(wake.s)
#         else:
#             # compute with zero padding to give desired frequency sampling
#             delta_f_natural = 1/(wake.s[-1]/c)
#             n_samples = int(len(wake.s) * delta_f_natural / delta_f)
#         f = rfftfreq(d=ds/c, n=n_samples)
#         Z = -np.array(rfft(wake.W, n=n_samples)) * ds/c

#         # compensate wake starting before 0
#         Z *= np.exp(-1j * 2*pi*f * wake.s[0]/c) 

#         if bunch_sigma is not None:

#             if cutoff_at_n_sigma is None:
#                 cutoff_index = len(f)
#             else:
#                 df = f[1] - f[0]
#                 cutoff_index = int(cutoff_at_n_sigma / (2*pi*bunch_sigma/c) / df)

#             f = f[:cutoff_index]
#             Z = Z[:cutoff_index]

#             Z /= get_gaussian_bunch_spectrum(f, bunch_sigma)

#         return Impedance(f, Z)


#     def save_txt(
#         self,
#         filename: Path | str,
#         frequency_unit: str = 'Hz',
#         fmt: str = '%.18e',
#         delimiter: str = '\t',
#         newline: str = '\n',
#         header: str | None = None,
#         comments = '#',
#         encoding: str | None = None,
#     ) -> None:
        
#         try:    
#             frequency_factor = FREQUENCY_UNITS[frequency_unit]
#         except KeyError:
#             raise ValueError(f'Unknown frequency unit: "{frequency_unit}"')
        
#         if header is None:
#             header = f'f ({frequency_unit}){delimiter}Re(Z) (Ohm){delimiter}Im(Z) (Ohm)'

#         np.savetxt(
#             fname=filename,
#             X=np.column_stack((self.f/frequency_factor, np.real(self.Z), np.imag(self.Z))),
#             fmt=fmt, delimiter=delimiter, newline=newline,
#             header=header, comments=comments, encoding=encoding
#         )


# class Wake:

#     def __init__(
#         self,
#         s: NDArray,
#         W: NDArray
#     ) -> None:

#         if not np.isrealobj(s):
#             raise ValueError('Distance values must be real-valued')

#         if not np.isrealobj(W):
#             raise ValueError('Wake potential/function values must be real-valued')
        
#         self.s = s
#         self.W = W


#     def __add__(self, other: Wake) -> Wake:
#         if not np.allclose(self.s, other.s):
#             raise ValueError('Distance samples do not match for Wake addition')
#         return Wake(self.s, self.W + other.W)
    

#     # implement right-side addition to support sum() function
#     def __radd__(self, other: Wake | int) -> Wake:
#         if other == 0:
#             return self
#         elif isinstance(other, Wake):
#             return self.__add__(other)
#         else:
#             raise TypeError('Unsupported type for addition with Wake')


#     def __mul__(self, scalar: float) -> Wake:
#         return Wake(self.s, self.W * scalar)


#     @classmethod
#     def from_cst_export(
#         cls,
#         filename: Path | str | list[str],
#     ) -> Wake:
        
#         if isinstance(filename, (Path, str)):
#             with open(filename) as fp:
#                 lines = fp.readlines()
#         elif isinstance(filename, list):
#             lines = filename
#         else:
#             raise TypeError('filename must be a path (pathlib.Path or str), or list of text lines (list[str])')
        
#         # get distance unit
#         re_match = None
#         for line in lines:
#             # match "s / mm", "s / cm", ...
#             re_match = re.search(
#                 r'"s \/ (' + '|'.join(LENGTH_UNITS.keys()) + r')"',
#                 line)
#             if re_match is None:
#                 continue
#             else:
#                 break

#         if re_match is None:
#             raise RuntimeError(f'Could not determine length unit in file "{filename}"')
#         try:    
#             length_factor = LENGTH_UNITS[re_match.group(1)]
#         except KeyError:
#             raise RuntimeError(f'Unknown length unit: "{re_match.group(1)}"')

#         raw = np.loadtxt(lines, comments='#')

#         return Wake(
#             raw[:,0] * length_factor, # m
#             raw[:,1] * 1e12 # V/C
#         )