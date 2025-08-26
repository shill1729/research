from itertools import combinations
from tda.ellipsoidal.solvers.scipy_solver import ellipsoidal_intersection
from tda.toydata.toydata import generate_toy_data


def build_ellipsoidal_cech(xs, A_list, eps, max_dim=2):
    """
    Constructs the ellipsoidal Čech complex up to dimension `max_dim`
    at scale `eps`, using the provided K‐function solvers.

    Parameters
    ----------
    xs : ndarray, shape (n, D)
        Point cloud coordinates (centers of the ellipsoids).
    A_list : ndarray, shape (n, D, D)
        Array of positive‐definite matrices defining each ellipsoid:
        the quadratic form A_list[i] * (x - xs[i]).
    eps : float
        Scale parameter: the radius (squared) at which to test intersections.
    max_dim : int, optional
        Maximum simplex dimension to build (0 = vertices, 1 = edges, etc.).

    Returns
    -------
    simplices : dict
        A mapping from dimension d (int) to a list of tuples, each tuple
        of vertex indices forming a d‐simplex in the ellipsoidal Čech complex.
    """
    n, D = xs.shape
    simplices = {d: [] for d in range(max_dim + 1)}

    # 0‐simplices: each point is a vertex
    simplices[0] = [(i,) for i in range(n)]

    # 1‐simplices: include edge (i, j) if the two ellipsoids intersect
    for i, j in combinations(range(n), 2):
        if ellipsoidal_intersection(eps, xs[[i, j]], A_list[[i, j]]):
            simplices[1].append((i, j))

    # Higher‐order simplices by checking k‐way intersection
    for dim in range(2, max_dim + 1):
        for simplex in combinations(range(n), dim + 1):
            # prune: all faces must already be present
            faces = combinations(simplex, dim)
            if all(tuple(face) in simplices[dim - 1] for face in faces):
                # test the intersection of the dim+1 ellipsoids
                idx = list(simplex)
                if ellipsoidal_intersection(eps, xs[idx], A_list[idx]):
                    simplices[dim].append(simplex)

    return simplices

# Example usage:
if __name__ == "__main__":
    from ae.toydata.curves import Parabola
    # TODO: use our synthetic data
    # load or generate your data:
    # xs is (n, D) array of points
    # A_list is (n, D, D) array of inverse‐covariance matrices
    # eps is your chosen scale parameter
    # max_dim is the top simplex dimension to build
    #
    # e.g.:
    # TODO: generate points
    curve = Parabola()
    xs, A_list, point_cloud = generate_toy_data(n=3, surface_or_curve=curve)
    eps = 0.5
    max_dim = 3

    # Build the complex
    complex_simplices = build_ellipsoidal_cech(xs, A_list, eps, max_dim)

    # Now `complex_simplices` can be fed into GUDHI's SimplexTree or any
    # other persistent‐homology library for further analysis.
    for d, sims in complex_simplices.items():
        print(f"{d}-simplices ({len(sims)}): {sims}")
