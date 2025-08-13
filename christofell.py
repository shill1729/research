from ae.symbolic.diffgeo import RiemannianManifold
import sympy as sp

theta, phi = sp.symbols("theta phi", real=True)

local_coordinates = sp.Matrix([theta, phi])

# Standard spherical coordinate chart for unit sphere
# x = sin(theta) * cos(phi)
# y = sin(theta) * sin(phi)
# z = cos(theta)
chart = sp.Matrix([
    sp.sin(theta) * sp.cos(phi),
    sp.sin(theta) * sp.sin(phi),
    sp.cos(theta)
])

manifold = RiemannianManifold(local_coordinates, chart)
print(manifold.christoffel_symbols())