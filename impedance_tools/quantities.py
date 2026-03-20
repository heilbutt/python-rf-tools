# necessary to type hinting class instance within itself for python<3.14
from __future__ import annotations

import numpy as np

from numpy.typing import NDArray


class Impedance:

    def __init__(
        self,
        f: NDArray,
        Z: NDArray
    ) -> None:

        if not np.isrealobj(f):
            raise ValueError('Frequency values must be real-valued')

        if not np.iscomplexobj(Z):
            raise ValueError('Impedance values must be complex-valued')
        
        self.f = f
        self.Z = Z


    def __add__(self, other: Impedance) -> Impedance:
        if not np.allclose(self.f, other.f):
            raise ValueError('Frequencies do not match for impedance addition')
        return Impedance(self.f, self.Z + other.Z)
    

    def __radd__(self, other: Impedance | int) -> Impedance:
        if other == 0:
            return self
        elif isinstance(other, Impedance):
            return self.__add__(other)
        else:
            raise TypeError('Unsupported type for addition with Impedance')


    def __sub__(self, other: Impedance) -> Impedance:
        if not np.allclose(self.f, other.f):
            raise ValueError('Frequencies do not match for impedance subtraction')
        return Impedance(self.f, self.Z - other.Z)


    def __mul__(self, scalar: float) -> Impedance:
        return Impedance(self.f, scalar * self.Z)
    

    def __rmul__(self, scalar: float) -> Impedance:
        return self * scalar
    

    def __truediv__(self, scalar: float) -> Impedance:
        return Impedance(self.f, self.Z / scalar)
    

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