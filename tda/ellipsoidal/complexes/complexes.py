import numpy as np
from itertools import combinations
from tda.ellipsoidal.solvers.scipy_solver import ellipsoidal_intersection


class EllipsoidalComplex:
    """
    Base class for ellipsoidal Čech and Vietoris–Rips complexes.
    """

    def __init__(self, points, A_matrices, eps):
        """
        Parameters:
        points: (n, d) array of point coordinates
        A_matrices: (n, d, d) array of positive definite matrices defining ellipsoidal shape
        eps: scalar scaling parameter (radius)
        """
        self.points = np.asarray(points)
        self.A_matrices = np.asarray(A_matrices)
        self.eps = float(eps)
        self.n_points = self.points.shape[0]
        self.dimension = self.points.shape[1]

        # Initialize complex storage
        self.vertices = [(i,) for i in range(self.n_points)]
        self.edges = []
        self.triangles = []
        self.tetrahedra = []

    def _check_simplex(self, simplex_indices):
        """
        Check whether the ellipsoids associated to the given indices intersect.
        Čech and VR subclasses override this to specialize behavior.
        """
        if len(simplex_indices) < 2:
            return True
        xs = self.points[list(simplex_indices)]
        A_subset = self.A_matrices[list(simplex_indices)]
        # TODO: swap in other methods once we have other solvers to time.
        return ellipsoidal_intersection(self.eps, xs, A_subset)

    def build_complex(self, max_dim=3):
        """
        Construct the simplicial complex up to the specified dimension.

        Stores the k-simplices in attributes:
        - self.edges         (1-simplices)
        - self.triangles     (2-simplices)
        - self.tetrahedra    (3-simplices)

        Parameters:
        max_dim: int (maximal dimension of simplices to include)
        """
        self.edges = []
        self.triangles = []
        self.tetrahedra = []

        targets = {1: self.edges, 2: self.triangles, 3: self.tetrahedra}
        for k in range(1, max_dim + 1):
            if k > 3:
                raise UserWarning("Maximum supported simplex dimension is 3.")
            for simplex in combinations(range(self.n_points), k + 1):
                if self._check_simplex(simplex):
                    targets[k].append(simplex)


class EllipsoidalVR(EllipsoidalComplex):
    """
    Ellipsoidal Vietoris–Rips complex:
    A k-simplex is included iff all (k+1 choose 2) ellipsoidal balls intersect pairwise.
    """

    def _check_simplex(self, simplex_indices):
        if len(simplex_indices) < 2:
            return True
        for i, j in combinations(simplex_indices, 2):
            xs = self.points[[i, j]]
            As = self.A_matrices[[i, j]]
            try:
                if not ellipsoidal_intersection(self.eps, xs, As):
                    return False
            except ValueError:
                return False
        return True

# TODO: Isn't this class redundant? It duplicates _check_simplex exactly from the base-class, no?
class EllipsoidalCech(EllipsoidalComplex):
    """
    Ellipsoidal Čech complex:
    A k-simplex is included iff the intersection of all k+1 ellipsoidal balls is nonempty.
    """

    def _check_simplex(self, simplex_indices):
        if len(simplex_indices) < 2:
            return True
        xs = self.points[list(simplex_indices)]
        As = self.A_matrices[list(simplex_indices)]
        return ellipsoidal_intersection(self.eps, xs, As)
