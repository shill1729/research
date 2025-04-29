import sympy as sp
import numpy as np
from abc import ABC, abstractmethod


# Abstract Base Class
class CurveBase(ABC):
    def __init__(self):
        """
        Abstract base class for curves.

        Attributes:
        -----------
        u : sympy.Symbol
            Symbol representing the parameter of the curve equation.
        """
        self.u = sp.symbols('u', real=True)

    def local_coords(self):
        """
        Get the local coordinate of the curve.

        Returns:
        --------
        sympy.Matrix
            Local coordinate [u].
        """
        return sp.Matrix([self.u])

    @abstractmethod
    def equation(self):
        """
        Abstract method to define the curve equation.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the curve.
        """
        pass

    @abstractmethod
    def bounds(self):
        """
        Abstract method to define the parameter bounds for the curve.

        Returns:
        --------
        list of tuples
            Bounds for the parameter u.
        """
        pass


# Curves of the form (u, f(u))
class Parabola(CurveBase):
    def __init__(self, a=2.):
        super().__init__()
        self.a = a

    def equation(self):
        return sp.Matrix([self.u,  (self.u/self.a)**2])

    def bounds(self):
        return [(-1., 1.)]

class Cubic(CurveBase):
    def __init__(self, a=1.05):
        super().__init__()
        self.a = a

    def equation(self):
        return sp.Matrix([self.u,  (self.u/self.a)**3])

    def bounds(self):
        return [(-1., 1.)]

class HalfEllipse(CurveBase):
    def __init__(self, a=2., b =2.):
        super().__init__()
        self.a = a
        self.b = b

    def equation(self):
        return sp.Matrix([self.u,  self.b*sp.sqrt(1-(self.u/self.a)**2)])

    def bounds(self):
        return [(-self.a, self.a)]


class SineCurve(CurveBase):
    def __init__(self, amplitude=1.0, frequency=1.0, phase=0.0):
        super().__init__()
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase

    def equation(self):
        return sp.Matrix([self.u, self.amplitude * sp.sin(self.frequency * self.u + self.phase)])

    def bounds(self):
        return [(-np.pi, np.pi)]


class EllipticCurve(CurveBase):
    def __init__(self, a=-1., b=1.):
        super().__init__()
        self.a = a
        self.b = b

    def equation(self):
        x = self.u
        argument = x**3 + self.a * x + self.b
        return sp.Matrix([x, sp.sqrt(argument)])

    def bounds(self):

        return [(-1.32473, 3.)]

class PinchedBellCurve(CurveBase):
    def __init__(self, a=-1., b=1.):
        super().__init__()
        self.a = a
        self.b = b

    def equation(self):
        x = self.u * (1-self.a * sp.exp(-self.b *(self.u**2-0.2)**2))
        y = sp.exp(-self.u**2/10)
        return sp.Matrix([x, y])

    def bounds(self):

        return [(-1.32473, 3.)]


class LogarithmicCurve(CurveBase):
    def __init__(self, a=1):
        super().__init__()
        self.a = a

    def equation(self):
        return sp.Matrix([self.u, self.a * sp.log(self.u)])

    def bounds(self):
        return [(0.1, 4)]  # Avoiding singularity at u=0


class ExponentialCurve(CurveBase):
    def __init__(self, a=1., b=1.):
        super().__init__()
        self.a = a
        self.b = b

    def equation(self):
        return sp.Matrix([self.u, self.a * sp.exp(self.b * self.u)])

    def bounds(self):
        return [(-2, 2)]


class RationalCurve(CurveBase):
    def __init__(self, a=1, b=1):
        super().__init__()
        self.a = a
        self.b = b

    def equation(self):
        return sp.Matrix([self.u, self.a / (1 + self.b * self.u**2)])

    def bounds(self):
        return [(-2, 2)]

class BellCurve(CurveBase):
    def __init__(self, a=1, b=1.):
        super().__init__()
        self.a = a
        self.b = b

    def equation(self):
        f = sp.exp(-self.u**2/(2*self.b**2))/sp.sqrt(sp.pi*self.b**2)
        return sp.Matrix([self.u, f])

    def bounds(self):
        return [(-1., 1.)]

# Example of usage
if __name__ == "__main__":
    curves = [
        Parabola(a=1),
        SineCurve(amplitude=1, frequency=1, phase=0),
        EllipticCurve(a=-1, b=1),
        LogarithmicCurve(a=2),
        ExponentialCurve(a=1, b=0.5),
        RationalCurve(a=1, b=1)
    ]

    for curve in curves:
        print(f"Curve: {curve.__class__.__name__}")
        print(f"Local Coordinates: {curve.local_coords()}")
        print(f"Equation: {curve.equation()}")
        print(f"Bounds: {curve.bounds()}")
