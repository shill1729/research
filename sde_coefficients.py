import sympy as sp
u, v = sp.symbols(names="u v", real=True)
# Local drifts to test
drift_zero = sp.Matrix([0, 0])
drift_circular = sp.Matrix([-sp.sin(u) + v ** 2 - 1, sp.exp(-v)])
harmonic_potential = - 0.5 * sp.Matrix([u, v])
morse_potential = sp.Matrix([-2 * u * sp.exp(-u ** 2 - v ** 2), -2 * v * sp.exp(-u ** 2 - v ** 2)])
double_well_potential = sp.Matrix([4 * u * (u ** 2 - 1), 2 * v])
lennard_jones_potential = sp.Matrix([12 * u / (u ** 2 + v ** 2) ** 7 - 24 * u / (u ** 2 + v ** 2) ** 13,
                                     12 * v / (u ** 2 + v ** 2) ** 7 - 24 * v / (u ** 2 + v ** 2) ** 13])


# Local diffusions to test:
diffusion_identity = sp.Matrix([[1, 0], [0, 1]])
diffusion_diagonal = sp.Matrix([[u, 0], [0, v]]) * 0.25
diffusion_circular = sp.Matrix([[5 + 0.5 * sp.cos(u), v],
                                [0, 0.2 + 0.1 * sp.sin(v)]])