import numpy as np

from typing import Any
from numpy.typing import NDArray


class XAxisMismatchError(Exception):
    pass


class UnequalSampleCountError(Exception):
    pass


class RealQuantity:

    def __init__(
        self,
        x: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any]]
    ) -> None:

        if not (x.ndim == 1 and np.isrealobj(x)):
            raise ValueError('X axis must be real-valued 1D numpy array')
        
        if not (y.ndim == 1 and np.isrealobj(y)):
            raise ValueError('Y axis must be real-valued 1D numpy array')
        
        if len(x) != len(y):
            raise UnequalSampleCountError(f'X and Y have unequal number of samples! (X: `{x}`, Y: `{y}`)')

        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=float)

    @property
    def n(self) -> int:
        return len(self._x)

    @property
    def x(self) -> NDArray[np.floating[Any]]:
        return self._x

    @x.setter
    def x(self, new_values: NDArray[np.floating[Any]]) -> None:
        
        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('X axis must be real-valued 1D numpy array')

        if not len(new_values) == self.n:
            raise UnequalSampleCountError(f'Changing number of X samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        
        self._x = np.asarray(new_values, dtype=float)

    @property
    def y(self) -> NDArray[np.floating[Any]]:
        return self._y
    
    @y.setter
    def y(self, new_values: NDArray[np.floating[Any]]) -> None:

        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('Y axis must be real-valued 1D numpy array')

        if not len(new_values) == self.n:
            raise UnequalSampleCountError(f'Changing number of Y samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        
        self._y = np.asarray(new_values, dtype=float)

    def __neg__(self) -> RealQuantity:
        return type(self)(self.x, -self.y)

    def __add__(self, other: RealQuantity | float) -> RealQuantity :
        
        if isinstance(other, RealQuantity):
            if not np.allclose(self.x, other.x):
                raise XAxisMismatchError('X values do not match for addition')
            return type(self)(self.x, self.y + other.y)
        
        if isinstance(other, (int, float)):
            return type(self)(self.x, self.y + other)
        
        return NotImplemented

    def __radd__(self, other: RealQuantity | float) -> RealQuantity:
        return self + other

    def __sub__(self, other: RealQuantity | float) -> RealQuantity:
        return self + (-other)
    
    def __rsub__(self, other: RealQuantity | float) -> RealQuantity:
        return other - self

    def __mul__(self, other: float) -> RealQuantity:
        
        if isinstance(other, (int, float)):
            return type(self)(self.x, other * self.y)
        
        return NotImplemented

    def __rmul__(self, other: float) -> RealQuantity:
        return self * other
    
    def __truediv__(self, other: float) -> RealQuantity:

        if isinstance(other, (int, float, complex)):
            return type(self)(self.x, self.y / other)
        
        return NotImplemented


class ComplexQuantity:

    def __init__(
        self,
        x: NDArray[np.floating[Any]],
        y: NDArray[np.floating[Any] | np.complexfloating[Any]]
    ) -> None:

        if not (x.ndim == 1 and np.isrealobj(x)):
            raise ValueError('X axis must be real-valued 1D numpy array')
        
        if not (y.ndim == 1):
            raise ValueError('Y axis must be real or complex-valued 1D numpy array')
        
        if len(x) != len(y):
            raise UnequalSampleCountError(f'X and Y have unequal number of samples! (X: `{x}`, Y: `{y}`)')

        self._x = np.asarray(x, dtype=float)
        self._y = np.asarray(y, dtype=complex)

    @property
    def n(self) -> int:
        return len(self._x)

    @property
    def x(self) -> NDArray[np.floating[Any]]:
        return self._x

    @x.setter
    def x(self, new_values: NDArray[np.floating[Any]]) -> None:
        
        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('X axis must be real-valued 1D numpy array')

        if not len(new_values) == self.n:
            raise UnequalSampleCountError(f'Changing number of X samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        
        self._x = np.asarray(new_values, dtype=float)

    @property
    def y(self) -> NDArray[np.complexfloating[Any]]:
        return self._y
    
    @y.setter
    def y(self, new_values: NDArray[np.floating[Any] | np.complexfloating[Any]]) -> None:

        if not (new_values.ndim == 1):
            raise ValueError('Y axis must be real or complex-valued 1D numpy array')

        if not len(new_values) == self.n:
            raise UnequalSampleCountError(f'Changing number of Y samples is not supported (current samples: `{self.n}`, new samples: `{len(new_values)}`). Create a new object instead')
        
        self._y = np.asarray(new_values, dtype=complex)

    @property
    def real(self) -> NDArray[np.floating[Any]]:
        return np.real(self.y)
    
    @real.setter
    def real(self, new_values: NDArray[np.floating[Any]]) -> None:
        
        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('New real part must be real-valued 1D numpy array')

        self.y = new_values + 1j*np.imag(self.y)
    
    @property
    def imag(self) -> NDArray[np.floating[Any]]:
        return np.imag(self.y)

    @imag.setter
    def imag(self, new_values: NDArray[np.floating[Any]]) -> None:
        
        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('New imaginary part must be real-valued 1D numpy array')
        
        self.y = np.real(self.y) + 1j*new_values

    @property
    def mag(self) -> NDArray[np.floating[Any]]:
        return np.abs(self.y)
    

    @mag.setter
    def mag(self, new_values: NDArray[np.floating[Any]]) -> None:
        
        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('New magnitude must be real-valued 1D numpy array')

        self.y = np.abs(new_values) * np.exp(1j*np.angle(self.y))

    @property
    def phase(self) -> NDArray[np.floating[Any]]:
        return np.angle(self.y)
    
    @phase.setter
    def phase(self, new_values: NDArray[np.floating[Any]]) -> None:
        
        if not (new_values.ndim == 1 and np.isrealobj(new_values)):
            raise ValueError('New phase must be real-valued 1D numpy array')

        self.y = np.abs(self.y) * np.exp(1j*new_values)

    def __neg__(self) -> ComplexQuantity:
        return type(self)(self.x, -self.y)

    def __add__(self, other: RealQuantity | ComplexQuantity | complex) -> ComplexQuantity :
        
        if isinstance(other, (RealQuantity, ComplexQuantity)):
            if not np.allclose(self.x, other.x):
                raise XAxisMismatchError('X values do not match for addition')
            return type(self)(self.x, self.y + other.y)
        
        if isinstance(other, (int, float, complex)):
            return type(self)(self.x, self.y + other)
        
        return NotImplemented

    def __radd__(self, other: RealQuantity | ComplexQuantity | complex) -> ComplexQuantity:
        return self + other

    def __sub__(self, other: RealQuantity | ComplexQuantity | complex) -> ComplexQuantity:
        return self + (-other)
    
    def __rsub__(self, other: RealQuantity | ComplexQuantity | complex) -> ComplexQuantity:
        return other - self

    def __mul__(self, other: complex) -> ComplexQuantity:
        
        if isinstance(other, (int, float, complex)):
            return type(self)(self.x, other * self.y)
        
        return NotImplemented

    def __rmul__(self, other: complex) -> ComplexQuantity:
        return self * other
    
    def __truediv__(self, other: complex) -> ComplexQuantity:

        if isinstance(other, (int, float, complex)):
            return type(self)(self.x, self.y / other)
        
        return NotImplemented