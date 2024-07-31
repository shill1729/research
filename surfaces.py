import sympy as sp
import numpy as np

u, v = sp.symbols(names="u v", real=True)
# R, r = sp.symbols(names="R r", positive=True)
# c1, c2 = sp.symbols(names="c1 c2", positive=True)
# sd = sp.symbols('sd', positive=True)
c1 = 2
c2 = 2
R = 2
r = 1
sd = 5
# First, define a dictionary with just the bounds
surface_bounds = {
    'quartic': [(-1, 1), (-1, 1)],
    'paraboloid': [(-1, 1), (-1, 1)],
    'sphere': [(0, np.pi), (0, 2 * np.pi)],
    'torus': [(0, 2 * np.pi), (0, 2 * np.pi)],
    'gaussian_bump': [(-3, 3), (-3, 3)],
    'hyperbolic_paraboloid': [(-2, 2), (-2, 2)],
    'hyperboloid': [(0, 2 * np.pi), (-2, 2)],
    'cylinder': [(0, 2 * np.pi), (-2, 2)],
    'mobius_strip': [(0, 2 * np.pi), (-1, 1)],
    'helicoid': [(-np.pi, np.pi), (-1, 1)],
    'dinis_surface': [(0, 2 * np.pi), (0.1, 0.9 * np.pi)],
    'cone': [(0, 1), (0, 2 * np.pi)],
    'hyperboloid_one_sheet': [(-1, 1), (0, 2 * np.pi)],
    'hyperboloid_two_sheets': [(1, 2), (0, 2 * np.pi)],
    'ellipsoid': [(0, np.pi), (0, 2 * np.pi)],
    'klein_bottle': [(0, np.pi), (0, 2 * np.pi)],
    'enneper_surface': [(-2, 2), (-2, 2)]
}

# Surfaces
quartic = sp.Matrix([u, v, (u/c1)**4-(v/c2)**2])
paraboloid = sp.Matrix([u, v, (u / c1) ** 2 + (v / c2) ** 2])
hyperbolic_paraboloid = sp.Matrix([u, v, (v / c2) ** 2 - (u / c1) ** 2])
hyperboloid = sp.Matrix([sp.cosh(v) * sp.cos(u), sp.cosh(v) * sp.sin(u), sp.sinh(v)])
sphere = sp.Matrix([sp.sin(u) * sp.cos(v), sp.sin(u) * sp.sin(v), sp.cos(u)])

torus = sp.Matrix([(R + r * sp.cos(u)) * sp.cos(v),
                   (R + r * sp.cos(u)) * sp.sin(v),
                   r * sp.sin(u)])

gaussian_bump = sp.Matrix([u, v, sp.exp(-(u ** 2 + v ** 2) / 2) / sp.sqrt(2 * sp.pi)])

cylinder = sp.Matrix([sp.cos(u), sp.sin(u), v])

mobius_strip = sp.Matrix([(1 + (v * sp.cos(u / 2)) / 2) * sp.cos(u),
                          (1 + (v * sp.cos(u / 2)) / 2) * sp.sin(u),
                          (v / 2) * sp.sin(u / 2)])

helicoid = sp.Matrix([v * sp.cos(3 * u), v * sp.sin(3 * u), u])

dinis_surface = sp.Matrix([sp.cos(u) * sp.sin(v),
                           sp.sin(u) * sp.cos(v),
                           u + sp.log(sp.tan(v / 2)) + sp.cos(v)])

cone = sp.Matrix([u * sp.cos(v), u * sp.sin(v), u])

hyperboloid_one_sheet = sp.Matrix([sp.cosh(u) * sp.cos(v),
                                   sp.cosh(u) * sp.sin(v),
                                   sp.sinh(u)])

hyperboloid_two_sheets = sp.Matrix([sp.sinh(u) * sp.cos(v),
                                    sp.sinh(u) * sp.sin(v),
                                    sp.cosh(u)])

ellipsoid = sp.Matrix([c1 * sp.sin(u) * sp.cos(v),
                       c2 * sp.sin(u) * sp.sin(v),
                       r * sp.cos(u)])

klein_bottle = sp.Matrix([(sp.cos(u) * (1 + sp.sin(u)) + v * sp.cos(u) / 2) * sp.cos(u),
                          (sp.cos(u) * (1 + sp.sin(u)) + v * sp.cos(u) / 2) * sp.sin(u),
                          v * sp.sin(u) / 2])

enneper_surface = sp.Matrix([u - (u ** 3) / 3 + u * v ** 2,
                             v - (v ** 3) / 3 + v * u ** 2,
                             u ** 2 - v ** 2])

