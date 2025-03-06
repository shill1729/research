import sympy as sp
from abc import ABC, abstractmethod
from ae.symbolic.diffgeo import RiemannianManifold


# Abstract Base Class
class DynamicsBase(ABC):
    def __init__(self):
        """
        Abstract base class for dynamics.

        Attributes:
        -----------
        u : sympy.Symbol
            Symbol representing the first variable of the dynamics.
        v : sympy.Symbol
            Symbol representing the second variable of the dynamics.
        """
        self.u, self.v = sp.symbols('u v', real=True)

    def local_coords(self):
        """
        Get the local coordinates.

        Returns:
        --------
        sympy.Matrix
            Local coordinates [u, v].
        """
        return sp.Matrix([self.u, self.v])

    @abstractmethod
    def drift(self, manifold):
        """
        Abstract method to define the drift vector.

        Returns:
        --------
        sympy.Matrix
            Drift vector.
        """
        pass

    @abstractmethod
    def diffusion(self, manifold):
        """
        Abstract method to define the diffusion matrix.

        Returns:
        --------
        sympy.Matrix
            Diffusion matrix.
        """
        pass

    @abstractmethod
    def name(self):
        "Name of the process"
        pass


# Specific Dynamics Implementations

class BrownianMotion(DynamicsBase):
    def drift(self, manifold=None):
        return sp.Matrix([0, 0])

    def diffusion(self, manifold=None):
        return sp.Matrix([[1, 0], [0, 1]])

    def name(self):
        return "bm"


class RiemannianBrownianMotion(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        return manifold.local_bm_drift()

    def diffusion(self, manifold: RiemannianManifold):
        return manifold.local_bm_diffusion()

    def name(self):
        return "rbm"


class LangevinDoubleWell(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        double_well_potential = sp.Matrix([
            4 * self.u * (self.u ** 2 - 1),
            2 * self.v
        ]) / 4
        return -0.5 * manifold.metric_tensor().inv() * double_well_potential + manifold.local_bm_drift()

    def diffusion(self, manifold: RiemannianManifold):
        return manifold.local_bm_diffusion()

    def name(self):
        return "langevin_doublewell"


class LangevinHarmonicOscillator(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        harmonic_potential = 0.1 * sp.Matrix([
            self.u,
            self.v
        ])
        return -0.5 * manifold.metric_tensor().inv() * harmonic_potential + manifold.local_bm_drift()

    def diffusion(self, manifold: RiemannianManifold):
        return manifold.local_bm_diffusion()

    def name(self):
        return "langevin_harmonic"


class LangevinGaussianWell(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        gaussian_well_potential = 5. * sp.Matrix([
            -2 * self.u * sp.exp(-self.u ** 2 - self.v ** 2),
            -2 * self.v * sp.exp(-self.u ** 2 - self.v ** 2)
        ])
        return -0.5 * manifold.metric_tensor().inv() * gaussian_well_potential + manifold.local_bm_drift()

    def diffusion(self, manifold: RiemannianManifold):
        return manifold.local_bm_diffusion()

    def name(self):
        return "langevin_gaussianbump"


class LangevinMorsePotential(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        morse_potential = 5. * sp.Matrix([
            2 * self.u * (1 - sp.exp(-self.u)) * sp.exp(-self.u),
            2 * self.v * (1 - sp.exp(-self.v)) * sp.exp(-self.v)
        ])
        return -0.5 * manifold.metric_tensor().inv() * morse_potential + manifold.local_bm_drift()

    def diffusion(self, manifold: RiemannianManifold):
        return manifold.local_bm_diffusion()

    def name(self):
        return "langevin_morse"


class LangevinGravitationalPotential(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        gravitational_potential = sp.Matrix([
            -self.u / (self.u ** 2 + self.v ** 2) ** (3 / 2),
            -self.v / (self.u ** 2 + self.v ** 2) ** (3 / 2)
        ])
        return -0.5 * manifold.metric_tensor().inv() * gravitational_potential + manifold.local_bm_drift()

    def diffusion(self, manifold: RiemannianManifold):
        return manifold.local_bm_diffusion()

    def name(self):
        return "langevin_gravity"


class LangevinChemicalReactionPotential(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        reaction_potential = sp.Matrix([
            -self.u / (1 + self.u ** 2 + self.v ** 2),
            -self.v / (1 + self.u ** 2 + self.v ** 2)
        ])
        return -0.5 * manifold.metric_tensor().inv() * reaction_potential + manifold.local_bm_drift()

    def diffusion(self, manifold: RiemannianManifold):
        return manifold.local_bm_diffusion()

    def name(self):
        return "langevin_chem"


class ArbitraryMotion(DynamicsBase):
    def drift(self, manifold=None):
        return sp.Matrix([
            -5 * (self.u - 0.5),
            5 * (self.u - 1)
        ]) / 6

    def diffusion(self, manifold=None):
        return sp.Matrix([
            [0.1 * sp.sin(self.u) + self.v ** 2, 0.05 * sp.cos(self.v) + self.u],
            [0.02 * self.u * self.v, 0.1 + 0.1 * self.v]
        ]) / 5

    def name(self):
        return "arbitrary1"


class ArbitraryMotion2(DynamicsBase):
    def drift(self, manifold=None):
        return sp.Matrix([
            -self.v,
            self.u
        ]) / 10

    def diffusion(self, manifold=None):
        return sp.Matrix([
            [0.1 * sp.sin(self.u) + self.v ** 2, 0.05 * sp.cos(self.v) + self.u],
            [0.02 * self.u * self.v, 0.1 + 0.1 * self.v]
        ]) / 10

    def name(self):
        return "arbitrary2"


class AnisotropicDynamics(DynamicsBase):
    def __init__(self):
        super().__init__()

    def drift(self, manifold=None):
        """
        Define the drift vector for the dynamics.

        Parameters:
        -----------
        manifold : RiemannianManifold, optional
            The manifold on which the dynamics are defined. Ignored in this implementation.

        Returns:
        --------
        sympy.Matrix
            Drift vector [-u(1 - u^2 - v^2), -v(1 - u^2 - v^2)].
        """
        drift = sp.Matrix([
            -self.u * (1 - self.u ** 2 - self.v ** 2),
            -self.v * (1 - self.u ** 2 - self.v ** 2)
        ])
        return drift

    def diffusion(self, manifold=None):
        """
        Define the diffusion matrix for the dynamics.

        Parameters:
        -----------
        manifold : RiemannianManifold, optional
            The manifold on which the dynamics are defined. Ignored in this implementation.

        Returns:
        --------
        sympy.Matrix
            Diffusion matrix [[1 + u, v], [u, 1 + v]].
        """
        diffusion = sp.Matrix([
            [1 + self.u, self.v],
            [self.u, 1 + self.v]
        ])
        return diffusion

    def name(self):
        return "anisotropic"
