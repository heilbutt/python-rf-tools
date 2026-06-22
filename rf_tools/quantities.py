from __future__ import annotations

import numpy as np

from typing import Any, TypeAlias
from numpy.typing import NDArray

RealArray: TypeAlias = NDArray[np.floating]
ComplexArray: TypeAlias = NDArray[np.complexfloating]


class XAxisMismatchError(Exception):
    pass


class UnequalSampleCountError(Exception):
    pass


class RealQuantity:

    def __init__(
        self,
        x: RealArray,
        value: RealArray
    ) -> None:

        if not (x.ndim == 1 and np.isrealobj(x)):
            raise ValueError('X axis must be real-valued 1D numpy array')
        
        if not (value.ndim == 1 and np.isrealobj(value)):
            raise ValueError('Y axis must be real-valued 1D numpy array')
        
        if len(x) != len(value):
            raise UnequalSampleCountError(f'X and Y have unequal number of samples! (X: `{len(x)}`, Y: `{len(value)}`)')

        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(value, dtype=float)

    @property
    def n(self) -> int:
        return len(self._x)

    @property
    def x(self) -> RealArray:
        return self._x

    @x.setter
    def x(self, new_values: RealArray) -> None:
        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('X axis must be real-valued 1D numpy array')
        if not len(new_values) == self.n:
            raise UnequalSampleCountError(f'Changing number of X samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        self._x = np.asarray(new_values, dtype=float)

    @property
    def value(self) -> RealArray:
        return self._y
    
    @value.setter
    def value(self, new_values: RealArray) -> None:
        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('Y axis must be real-valued 1D numpy array')
        if not len(new_values) == self.n:
            raise UnequalSampleCountError(f'Changing number of Y samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        self._y = np.asarray(new_values, dtype=float)

    def __neg__(self) -> RealQuantity:
        return type(self)(self.x, -self.value)

    def __add__(self, other: RealQuantity | float) -> RealQuantity :
        if isinstance(other, RealQuantity):
            if not np.allclose(self.x, other.value):
                raise XAxisMismatchError('X values do not match for addition')
            return type(self)(self.x, self.value + other.value)
        if isinstance(other, (int, float)):
            return type(self)(self.x, self.value + other)
        
        return NotImplemented

    def __radd__(self, other: RealQuantity | float) -> RealQuantity:
        return self + other

    def __sub__(self, other: RealQuantity | float) -> RealQuantity:
        return self + (-other)
    
    def __rsub__(self, other: RealQuantity | float) -> RealQuantity:
        return other - self

    def __mul__(self, other: float) -> RealQuantity:
        if isinstance(other, (int, float)):
            return type(self)(self.x, other * self.value)
        return NotImplemented

    def __rmul__(self, other: float) -> RealQuantity:
        return self * other
    
    def __truediv__(self, other: float) -> RealQuantity:
        if isinstance(other, (int, float, complex)):
            return type(self)(self.x, self.value / other)
        return NotImplemented

    def __len__(self) -> int:
        return self.n
    
    def as_tuple(self) -> tuple[RealArray, RealArray]:
        return self.x, self.value
    
    def interpolate_to(self, new_x: RealArray) -> RealQuantity:
        new_values = np.interp(x=new_x, xp=self.x, fp=self.value)
        return type(self)(new_x, new_values)


class ComplexQuantity:

    def __init__(
        self,
        x: RealArray,
        value: RealArray | ComplexArray
    ) -> None:

        if not (x.ndim == 1 and np.isrealobj(x)):
            raise ValueError('`x` axis must be real-valued 1D numpy array')
        
        if not (value.ndim == 1):
            raise ValueError('`value` axis must be real or complex-valued 1D numpy array')
        
        if len(x) != len(value):
            raise UnequalSampleCountError(f'`x` and `value` have unequal number of samples! (`x`: {len(x)}, `value`: {len(value)})')
        
        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(value, dtype=complex)

    @property
    def n(self) -> int:
        return len(self._x)

    @property
    def x(self) -> RealArray:
        return self._x

    @x.setter
    def x(self, new_values: RealArray) -> None:
        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('X axis must be real-valued 1D numpy array')
        if not len(new_values) == self.n:
            raise UnequalSampleCountError(f'Changing number of X samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        self._x = np.asarray(new_values, dtype=float)

    @property
    def value(self) -> ComplexArray:
        return self._y
    
    @value.setter
    def value(self, new_values: NDArray[np.floating[Any] | np.complexfloating[Any]]) -> None:
        if not (new_values.ndim == 1):
            raise ValueError('Y axis must be real or complex-valued 1D numpy array')
        if not len(new_values) == self.n:
            raise UnequalSampleCountError(f'Changing number of Y samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        self._y = np.asarray(new_values, dtype=complex)

    @property
    def real(self) -> RealArray:
        return np.real(self.value)
    
    @real.setter
    def real(self, new_values: RealArray) -> None:
        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('New real part must be real-valued 1D numpy array')
        self.value = new_values + 1j*np.imag(self.value)
    
    @property
    def imag(self) -> RealArray:
        return np.imag(self.value)

    @imag.setter
    def imag(self, new_values: RealArray) -> None:
        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('New imaginary part must be real-valued 1D numpy array')
        self.value = np.real(self.value) + 1j*new_values

    @property
    def mag(self) -> RealArray:
        return np.abs(self.value)
    
    @mag.setter
    def mag(self, new_values: RealArray) -> None:
        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('New magnitude must be real-valued 1D numpy array')
        self.value = np.abs(new_values) * np.exp(1j*np.angle(self.value))

    @property
    def phase(self) -> RealArray:
        return np.angle(self.value)
    
    @phase.setter
    def phase(self, new_values: RealArray) -> None:
        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('New phase must be real-valued 1D numpy array')
        self.value = np.abs(self.value) * np.exp(1j*new_values)

    def __neg__(self) -> ComplexQuantity:
        return type(self)(self.x, -self.value)

    def __add__(self, other: RealQuantity | ComplexQuantity | complex) -> ComplexQuantity :
        if isinstance(other, (RealQuantity, ComplexQuantity)):
            if not np.allclose(self.x, other.x):
                raise XAxisMismatchError('X values do not match for addition')
            return type(self)(self.x, self.value + other.value)
        if isinstance(other, (int, float, complex)):
            return type(self)(self.x, self.value + other)
        
        return NotImplemented

    def __radd__(self, other: RealQuantity | ComplexQuantity | complex) -> ComplexQuantity:
        return self + other

    def __sub__(self, other: RealQuantity | ComplexQuantity | complex) -> ComplexQuantity:
        return self + (-other)
    
    def __rsub__(self, other: RealQuantity | ComplexQuantity | complex) -> ComplexQuantity:
        return other - self

    def __mul__(self, other: complex) -> ComplexQuantity:
        
        if isinstance(other, (int, float, complex)):
            return type(self)(self.x, other * self.value)
        
        return NotImplemented

    def __rmul__(self, other: complex) -> ComplexQuantity:
        return self * other
    
    def __truediv__(self, other: complex) -> ComplexQuantity:

        if isinstance(other, (int, float, complex)):
            return type(self)(self.x, self.value / other)
        
        return NotImplemented
    
    def __len__(self) -> int:
        return self.n
    
    def as_tuple(self) -> tuple[RealArray, ComplexArray]:
        return self.x, self.value
    
    def interpolate_to(self, new_x: RealArray) -> ComplexQuantity:
        new_values = np.interp(x=new_x, xp=self.x, fp=self.value)
        return type(self)(new_x, new_values)


def normalize(
    array: RealArray,
    minimum: float = 0,
    maximum: float = 1
) -> RealArray:

    return minimum + (
        (array - min(array))
        * (maximum - minimum)
        / (max(array) - min(array))
    )