import sympy as sp
import numpy as np
from abc import ABC, abstractmethod


# Abstract Base Class
class SurfaceBase(ABC):
    def __init__(self):
        """
        Abstract base class for surfaces.

        Attributes:
        -----------
        u : sympy.Symbol
            Symbol representing the first parameter of the surface equations.
        v : sympy.Symbol
            Symbol representing the second parameter of the surface equations.
        """
        self.u, self.v = sp.symbols('u v', real=True)

    def local_coords(self):
        """
        Get the local coordinates of the surface.

        Returns:
        --------
        sympy.Matrix
            Local coordinates [u, v].
        """
        return sp.Matrix([self.u, self.v])

    @abstractmethod
    def equation(self):
        """
        Abstract method to define the surface equation.

        Returns:
        --------
        sympy.Matrix
            Parametric equation of the surface.
        """
        pass

    @abstractmethod
    def bounds(self):
        """
        Abstract method to define the parameter bounds for the surface.

        Returns:
        --------
        list of tuples
            Bounds for the parameters u and v.
        """
        pass


# Surfaces of the form (x(u,v), y(u,v), z(u,v))
class Sphere(SurfaceBase):
    def equation(self):
        return sp.Matrix([
            sp.sin(self.u) * sp.cos(self.v),
            sp.sin(self.u) * sp.sin(self.v),
            sp.cos(self.u)
        ])

    def bounds(self):
        return [(0, np.pi), (0, 2 * np.pi)]


class Cylinder(SurfaceBase):
    def equation(self):
        return sp.Matrix([sp.cos(self.u), sp.sin(self.u), self.v])

    def bounds(self):
        return [(0, 2 * np.pi), (-2, 2)]


class Torus(SurfaceBase):
    def __init__(self, R=1, r=0.5):
        super().__init__()
        self.R = R
        self.r = r

    def equation(self):
        return sp.Matrix([
            (self.R + self.r * sp.cos(self.u)) * sp.cos(self.v),
            (self.R + self.r * sp.cos(self.u)) * sp.sin(self.v),
            self.r * sp.sin(self.u)
        ])

    def bounds(self):
        return [(0, 2 * np.pi), (0, 2 * np.pi)]


# Surfaces of the form $(x,y, f(x,y))$
class WaveSurface(SurfaceBase):
    def __init__(self, amplitude=1.0, frequency_u=1.0, frequency_v=0.0, phase=0.0):
        super().__init__()
        self.amplitude = amplitude
        self.frequency_u = frequency_u
        self.frequency_v = frequency_v
        self.phase = phase

    def equation(self):
        # Create a 2D wave pattern
        z = self.amplitude * sp.sin(self.frequency_u * self.u + self.frequency_v * self.v + self.phase)
        return sp.Matrix([self.u, self.v, z])

    def bounds(self):
        return [(-2 * np.pi, 2 * np.pi), (-2 * np.pi, 2 * np.pi)]


class Paraboloid(SurfaceBase):
    def __init__(self, a=6, b=6):
        super().__init__()
        self.a = a
        self.b = b

    def equation(self):
        fuv = (self.u / self.a) ** 2 + (self.v / self.b) ** 2
        return sp.Matrix([self.u, self.v, fuv])

    def bounds(self):
        return [(-1., 1.), (-1., 1.)]


class ProductSurface(SurfaceBase):
    def __init__(self, a=4):
        super().__init__()
        self.a = a

    def equation(self):
        fuv = self.u * self.v / self.a
        return sp.Matrix([self.u, self.v, fuv])

    def bounds(self):
        return [(-2., 2.), (-2., 2.)]


class HyperbolicParaboloid(SurfaceBase):
    def __init__(self, a=4, b=4):
        super().__init__()
        self.a = a
        self.b = b

    def equation(self):
        fuv = (self.v / self.b) ** 2 - (self.u / self.a) ** 2
        return sp.Matrix([self.u, self.v, fuv])

    def bounds(self):
        return [(-2., 2.), (-2., 2.)]


class SpherePatch(SurfaceBase):
    def equation(self):
        return sp.Matrix([
            self.u,
            self.v,
            sp.sqrt(1 - self.u ** 2 - self.v ** 2)
        ])

    def bounds(self):
        # (0, 0, 1), (0, sqrt(0.8), sqrt(1-0.8))
        return [(0, 0.5), (0, np.sqrt(0.5))]


class QuarticMinusCubic(SurfaceBase):
    def __init__(self, a=1, b=1):
        super().__init__()
        self.a = a
        self.b = b

    def equation(self):
        fuv = (self.u / self.a) ** 4 - (self.v / self.b) ** 3
        return sp.Matrix([self.u, self.v, fuv])

    def bounds(self):
        return [(-2., 2.), (-2., 2.)]


class RationalSurface(SurfaceBase):
    def __init__(self, a=1, b=1):
        super().__init__()
        self.a = a
        self.b = b

    def equation(self):
        fuv = (self.u + self.v) / (1 + self.u ** 2 + self.v ** 2)
        return sp.Matrix([self.u, self.v, fuv])

    def bounds(self):
        return [(-2., 2.), (-2., 2.)]


# Example of usage
if __name__ == "__main__":
    surfaces = [
        Cylinder(),
        Torus(R=3, r=1),
        Paraboloid(a=2, b=2),
        HyperbolicParaboloid(a=1, b=1),
        Sphere(),
        QuarticMinusCubic(a=1, b=1)
    ]

    for surface in surfaces:
        print(f"Surface: {surface.__class__.__name__}")
        print(f"Local Coordinates: {surface.local_coords()}")
        print(f"Equation: {surface.equation()}")
        print(f"Bounds: {surface.bounds()}")
