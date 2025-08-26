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


# Specific Dynamics Implementations
class BrownianMotion(DynamicsBase):
    def drift(self, manifold=None):
        return sp.Matrix([0, 0])

    def diffusion(self, manifold=None):
        return sp.Matrix([[1, 0], [0, 1]])


class RiemannianBrownianMotion(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        return manifold.local_bm_drift()

    def diffusion(self, manifold: RiemannianManifold):
        return manifold.local_bm_diffusion()


class LangevinHarmonicOscillator(DynamicsBase):

    def __init__(self, temperature=1., target_x=0.9, target_y=0.):
        """

        :param temperature:
        :param target_x:
        :param target_y:
        """
        super().__init__()
        self.temperature = temperature
        self.inverse_temp = 1 / temperature
        self.volatility = 1.
        self.target_x = target_x
        self.target_y = target_y

    def drift(self, manifold: RiemannianManifold):
        # Define a potential based on the dimension
        if manifold.local_coordinates.shape[0] == 2:
            harmonic_potential = sp.Matrix([
                self.u - self.target_x,
                self.v - self.target_y
            ])
        elif manifold.local_coordinates.shape[0] == 1:
            harmonic_potential = sp.Matrix([
                self.u - self.target_x
            ])
        else:
            raise NotImplementedError("Only intrinsic dimensions 2 and 1 are implemented")
        # Collect Riemannian drift plus potential drift
        manifold_drift = manifold.local_bm_drift()
        manifold_potential = -0.5 * manifold.metric_tensor().inv() * harmonic_potential
        drift_term = manifold_potential + manifold_drift * self.volatility
        return drift_term

    def diffusion(self, manifold: RiemannianManifold):
        manifold_diffusion = manifold.local_bm_diffusion()
        return manifold_diffusion * self.volatility


class LangevinDoubleWell(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        if manifold.local_coordinates.shape[0] == 2:
            xnorm_sq = self.u**2+self.v**2
            double_well_potential = 2*sp.Matrix([
                (4*xnorm_sq - 1)*self.u,
                (4*xnorm_sq - 1)*self.v
            ])
        elif manifold.local_coordinates.shape[0] == 1:
            double_well_potential = sp.Matrix([self.u * (4*self.u ** 2 - 1)])
        else:
            raise NotImplementedError("Only intrinsic dimensions 2 and 1 are implemented")
        manifold_drift = manifold.local_bm_drift()
        potential = manifold.metric_tensor().inv() * double_well_potential
        return -0.5 * potential + manifold_drift

    def diffusion(self, manifold: RiemannianManifold):
        return manifold.local_bm_diffusion()


class LangevinGaussianWell(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        gaussian_well_potential = 5. * sp.Matrix([
            -2 * self.u * sp.exp(-self.u ** 2 - self.v ** 2),
            -2 * self.v * sp.exp(-self.u ** 2 - self.v ** 2)
        ])
        return -0.5 * manifold.metric_tensor().inv() * gaussian_well_potential + manifold.local_bm_drift()

    def diffusion(self, manifold: RiemannianManifold):
        return manifold.local_bm_diffusion()


class LangevinMorsePotential(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        morse_potential = 5. * sp.Matrix([
            2 * self.u * (1 - sp.exp(-self.u)) * sp.exp(-self.u),
            2 * self.v * (1 - sp.exp(-self.v)) * sp.exp(-self.v)
        ])
        return -0.5 * manifold.metric_tensor().inv() * morse_potential + manifold.local_bm_drift()

    def diffusion(self, manifold: RiemannianManifold):
        return manifold.local_bm_diffusion()


class LangevinGravitationalPotential(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        gravitational_potential = sp.Matrix([
            -self.u / (self.u ** 2 + self.v ** 2) ** (3 / 2),
            -self.v / (self.u ** 2 + self.v ** 2) ** (3 / 2)
        ])
        return -0.5 * manifold.metric_tensor().inv() * gravitational_potential + manifold.local_bm_drift()

    def diffusion(self, manifold: RiemannianManifold):
        return manifold.local_bm_diffusion()


class LangevinChemicalReactionPotential(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        reaction_potential = sp.Matrix([
            -self.u / (1 + self.u ** 2 + self.v ** 2),
            -self.v / (1 + self.u ** 2 + self.v ** 2)
        ])
        return -0.5 * manifold.metric_tensor().inv() * reaction_potential + manifold.local_bm_drift()

    def diffusion(self, manifold: RiemannianManifold):
        return manifold.local_bm_diffusion()


class ArbitraryMotion(DynamicsBase):
    def drift(self, manifold=None):
        return sp.Matrix([
            -(self.u - 0.5),
             (self.u - 0.5*self.v**2/10)
        ]) / 2

    def diffusion(self, manifold=None):
        return sp.Matrix([
            [0.1 * sp.sin(self.u) + self.v ** 2, 0.05 * sp.cos(self.v) + self.u],
            [0.02 * self.u * self.v, 0.1 + 0.1 * self.v]
        ]) / 3


class ArbitraryMotion2(DynamicsBase):
    def drift(self, manifold=None):
        return sp.Matrix([
            -self.v,
            self.u
        ]) / (1 + self.v ** 2 + self.u ** 2)

    def diffusion(self, manifold=None):
        return sp.Matrix([
            [0.1 * sp.sin(self.u) + self.v ** 2, 0.05 * sp.cos(self.v) + self.u],
            [0.02 * self.u * self.v, 0.1 + 0.1 * self.v]
        ]) / 5


class ArbitraryMotion1D(DynamicsBase):
    def drift(self, manifold=None):
        return sp.Matrix([
            -self.u/ (1 + self.u ** 2)
        ])

    def diffusion(self, manifold=None):
        return sp.Matrix([
            [0.1 * sp.sin(self.u) + 0.02 * self.u ** 2]
        ]) / 5

class AnisotropicSDE(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        u, v = self.u, self.v

        # Define anisotropic drift a = (a^1, a^2)
        a = sp.Matrix([
            -2 * u + v,  # Example: Swirling motion or gradient flow
            -2 * v
        ])

        return -0.5 * manifold.metric_tensor().inv() * a + manifold.local_bm_drift()

    def diffusion(self, manifold: RiemannianManifold):
        u, v = self.u, self.v

        # Define anisotropic diffusion coefficient b (not necessarily isotropic)
        b = sp.Matrix([
            [1, 0],  # Stronger diffusion along u
            [0, sp.exp(-u)]  # Exponential growth in v-direction
        ])/10

        return manifold.local_bm_diffusion() * b


class AnisotropicSDE2(DynamicsBase):
    def drift(self, manifold: RiemannianManifold):
        u, v = self.u, self.v

        # Define anisotropic drift a = (a^1, a^2)
        a = sp.Matrix([
            -0.2 * sp.cos(u+v)*u*v,
            -0.5 * sp.sin(v**2)
        ])

        return a

    def diffusion(self, manifold: RiemannianManifold):
        u, v = self.u, self.v

        # Define anisotropic diffusion coefficient b (not necessarily isotropic)
        b = sp.Matrix([
            [sp.exp(-v), 0],  # Stronger diffusion along u
            [0, sp.exp(u)]  # Exponential growth in v-direction
        ])

        return b / 2.

