import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv, det
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
import warnings


class EllipsoidIntersectionOptimizer:
    """
    Novel algorithm for minimizing K(λ) over the probability simplex Δ^k
    for ellipsoidal ball intersection problems.

    This implementation combines:
    1. Adaptive projected gradient descent with momentum
    2. Geometric insights from KKT conditions
    3. Smart initialization using centroid analysis
    4. Boundary exploration for degenerate cases
    """

    def __init__(self, points: np.ndarray, precision_matrices: List[np.ndarray],
                 epsilon: float = 1.0, tolerance: float = 1e-8):
        """
        Initialize the optimizer.

        Args:
            points: Array of shape (k, D) containing k points in D-dimensional space
            precision_matrices: List of k precision matrices A_i^{-1}
            epsilon: Radius parameter for ellipsoidal balls
            tolerance: Convergence tolerance
        """
        self.points = np.array(points)
        self.k, self.D = self.points.shape
        self.A_inv = [np.array(A) for A in precision_matrices]
        self.epsilon = epsilon
        self.tol = tolerance

        # Validate inputs
        assert len(self.A_inv) == self.k, "Number of precision matrices must match number of points"
        for i, A in enumerate(self.A_inv):
            assert A.shape == (self.D, self.D), f"Precision matrix {i} has wrong shape"
            assert np.allclose(A, A.T), f"Precision matrix {i} is not symmetric"
            try:
                np.linalg.cholesky(A)
            except np.linalg.LinAlgError:
                raise ValueError(f"Precision matrix {i} is not positive definite")

    def compute_B_and_m(self, lam: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute B(λ)^{-1} and m(λ) as defined in the paper."""
        # B(λ)^{-1} = Σ λ_i A_i^{-1}
        B_inv = sum(lam[i] * self.A_inv[i] for i in range(self.k))

        # B(λ) = (B(λ)^{-1})^{-1}
        B = inv(B_inv)

        # m(λ) = B(λ) Σ λ_i A_i^{-1} x_i
        weighted_sum = sum(lam[i] * self.A_inv[i] @ self.points[i] for i in range(self.k))
        m = B @ weighted_sum

        return B_inv, B, m

    def compute_C(self, lam: np.ndarray, B_inv: np.ndarray, m: np.ndarray) -> float:
        """Compute C(λ) as defined in Lemma 4."""
        # C(λ) = Σ λ_i x_i^T A_i^{-1} x_i - m(λ)^T B(λ)^{-1} m(λ)
        first_term = sum(lam[i] * self.points[i].T @ self.A_inv[i] @ self.points[i]
                         for i in range(self.k))
        second_term = m.T @ B_inv @ m
        return first_term - second_term

    def compute_K(self, lam: np.ndarray) -> float:
        """Compute K_ε(λ) = ε² - C(λ)."""
        B_inv, B, m = self.compute_B_and_m(lam)
        C = self.compute_C(lam, B_inv, m)
        return self.epsilon ** 2 - C

    def compute_gradient(self, lam: np.ndarray) -> np.ndarray:
        """Compute gradient of K(λ) using Lemma 6."""
        B_inv, B, m = self.compute_B_and_m(lam)

        # ∂K/∂λ_j = -||m(λ) - x_j||²_{A_j^{-1}}
        grad = np.zeros(self.k)
        for j in range(self.k):
            diff = m - self.points[j]
            grad[j] = -diff.T @ self.A_inv[j] @ diff

        return grad

    def project_simplex(self, x: np.ndarray) -> np.ndarray:
        """Project point x onto probability simplex Δ^k."""
        # Use efficient projection algorithm
        x_sorted = np.sort(x)[::-1]  # Sort in descending order

        # Find the threshold
        cumsum = np.cumsum(x_sorted)
        ind = np.arange(1, len(x) + 1)
        cond = x_sorted - (cumsum - 1) / ind > 0

        if np.any(cond):
            rho = np.where(cond)[0][-1]
            theta = (cumsum[rho] - 1) / (rho + 1)
        else:
            theta = (cumsum[-1] - 1) / len(x)

        return np.maximum(x - theta, 0)

    def smart_initialization(self) -> np.ndarray:
        """
        Smart initialization based on geometric insights.
        Try multiple strategies and pick the best.
        """
        candidates = []

        # Strategy 1: Uniform distribution
        candidates.append(np.ones(self.k) / self.k)

        # Strategy 2: Based on distances to centroid
        centroid = np.mean(self.points, axis=0)
        distances = np.array([np.linalg.norm(self.points[i] - centroid)
                              for i in range(self.k)])
        # Inverse distance weighting
        if np.any(distances > 0):
            weights = 1 / (distances + 1e-10)
            candidates.append(weights / np.sum(weights))

        # Strategy 3: Based on volume of ellipsoids
        volumes = np.array([1 / np.sqrt(det(self.A_inv[i]) + 1e-10)
                            for i in range(self.k)])
        candidates.append(volumes / np.sum(volumes))

        # Strategy 4: Random perturbations
        for _ in range(3):
            random_weights = np.random.exponential(1, self.k)
            candidates.append(random_weights / np.sum(random_weights))

        # Evaluate all candidates and return the best
        best_lam = candidates[0]
        best_value = self.compute_K(best_lam)

        for lam in candidates[1:]:
            try:
                value = self.compute_K(lam)
                if value < best_value:
                    best_value = value
                    best_lam = lam
            except:
                continue

        return best_lam

    def adaptive_projected_gradient(self, max_iter: int = 1000,
                                    initial_lr: float = 0.1) -> Tuple[np.ndarray, float, dict]:
        """
        Adaptive projected gradient descent with momentum.
        """
        lam = self.smart_initialization()
        lr = initial_lr
        momentum = 0.9
        velocity = np.zeros(self.k)

        history = {'objective': [], 'lambda': [], 'lr': []}

        for iteration in range(max_iter):
            # Compute gradient
            try:
                grad = self.compute_gradient(lam)
                obj_val = self.compute_K(lam)
            except np.linalg.LinAlgError:
                # Handle numerical issues
                lr *= 0.5
                if lr < 1e-12:
                    break
                continue

            # Store history
            history['objective'].append(obj_val)
            history['lambda'].append(lam.copy())
            history['lr'].append(lr)

            # Adaptive learning rate based on gradient norm
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 10:
                lr *= 0.8
            elif grad_norm < 0.1:
                lr *= 1.1

            # Momentum update
            velocity = momentum * velocity - lr * grad

            # Projected gradient step
            lam_new = self.project_simplex(lam + velocity)

            # Check convergence
            if np.linalg.norm(lam_new - lam) < self.tol:
                lam = lam_new
                break

            # Line search with backtracking
            alpha = 1.0
            lam_candidate = lam_new
            obj_candidate = self.compute_K(lam_candidate)

            # Simple backtracking
            while alpha > 1e-8 and obj_candidate > obj_val + 1e-4 * alpha * grad.T @ (lam_candidate - lam):
                alpha *= 0.5
                lam_candidate = self.project_simplex(lam + alpha * velocity)
                try:
                    obj_candidate = self.compute_K(lam_candidate)
                except:
                    obj_candidate = float('inf')

            if alpha > 1e-8:
                lam = lam_candidate
            else:
                # Reset if line search fails
                velocity *= 0.1
                lr *= 0.5

        final_obj = self.compute_K(lam)

        return lam, final_obj, history

    def boundary_exploration(self, lam_interior: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Explore boundary of simplex for potentially better solutions.
        This leverages the KKT conditions insight that optimal solutions
        often lie on the boundary.
        """
        best_lam = lam_interior.copy()
        best_obj = self.compute_K(lam_interior)

        # Try fixing each coordinate to 0 and optimizing over remaining simplex
        for i in range(self.k):
            if lam_interior[i] > 0.1:  # Only try if coordinate is reasonably large
                # Create reduced problem
                indices = [j for j in range(self.k) if j != i]
                if len(indices) <= 1:
                    continue

                # Initialize reduced lambda
                lam_reduced = lam_interior[indices]
                lam_reduced = lam_reduced / np.sum(lam_reduced)

                # Optimize on this face
                try:
                    result = self._optimize_on_face(indices, lam_reduced)
                    if result[1] < best_obj:
                        # Reconstruct full lambda
                        lam_full = np.zeros(self.k)
                        lam_full[indices] = result[0]
                        best_lam = lam_full
                        best_obj = result[1]
                except:
                    continue

        return best_lam, best_obj

    def _optimize_on_face(self, active_indices: List[int],
                          lam_reduced: np.ndarray) -> Tuple[np.ndarray, float]:
        """Optimize K(λ) on a face of the simplex."""
        # Simple gradient descent on the face
        for _ in range(100):
            # Construct full lambda
            lam_full = np.zeros(self.k)
            lam_full[active_indices] = lam_reduced

            # Compute gradient
            grad_full = self.compute_gradient(lam_full)
            grad_reduced = grad_full[active_indices]

            # Project gradient to simplex tangent space
            grad_projected = grad_reduced - np.mean(grad_reduced)

            # Update
            lam_reduced_new = self.project_simplex(lam_reduced - 0.01 * grad_projected)

            if np.linalg.norm(lam_reduced_new - lam_reduced) < self.tol:
                break
            lam_reduced = lam_reduced_new

        lam_full = np.zeros(self.k)
        lam_full[active_indices] = lam_reduced
        return lam_reduced, self.compute_K(lam_full)

    def minimize(self, max_iter: int = 1000, explore_boundary: bool = True) -> dict:
        """
        Main optimization routine combining multiple strategies.

        Returns:
            Dictionary containing optimal λ, minimum value, and optimization details
        """
        print(f"Minimizing K(λ) for {self.k} ellipsoids in {self.D}D space...")

        # Phase 1: Adaptive projected gradient
        print("Phase 1: Adaptive projected gradient descent...")
        lam_opt, obj_opt, history = self.adaptive_projected_gradient(max_iter)

        # Phase 2: Boundary exploration (if requested)
        if explore_boundary and obj_opt > -self.epsilon ** 2:  # Only if not clearly infeasible
            print("Phase 2: Boundary exploration...")
            lam_boundary, obj_boundary = self.boundary_exploration(lam_opt)
            if obj_boundary < obj_opt:
                lam_opt = lam_boundary
                obj_opt = obj_boundary
                print(f"  Boundary exploration improved objective: {obj_opt:.6f}")

        # Phase 3: Final refinement with scipy
        print("Phase 3: Final refinement...")
        try:
            constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
            bounds = [(0, 1) for _ in range(self.k)]

            result = minimize(self.compute_K, lam_opt, method='SLSQP',
                              bounds=bounds, constraints=constraints,
                              options={'ftol': self.tol})

            if result.success and result.fun < obj_opt:
                lam_opt = result.x
                obj_opt = result.fun
                print(f"  Scipy refinement improved objective: {obj_opt:.6f}")
        except:
            print("  Scipy refinement failed, using our solution")

        # Compute final solution details
        B_inv, B, m = self.compute_B_and_m(lam_opt)

        # Check KKT conditions
        grad = self.compute_gradient(lam_opt)
        alpha = np.max(grad)  # Should be constant for active components

        # Identify active constraints
        active_indices = np.where(lam_opt > self.tol)[0]

        print(f"\nOptimization completed:")
        print(f"  Minimum K(λ*) = {obj_opt:.8f}")
        print(f"  Active components: {len(active_indices)}/{self.k}")
        print(
            f"  Intersection status: {'Non-empty' if obj_opt > 0 else 'Empty' if obj_opt < -self.tol else 'Boundary'}")

        return {
            'lambda_optimal': lam_opt,
            'objective_value': obj_opt,
            'centroid': m,
            'precision_matrix': B_inv,
            'active_indices': active_indices,
            'intersection_exists': obj_opt > -self.tol,
            'history': history,
            'kkt_multiplier': alpha
        }


# Example usage and testing
def test_algorithm():
    """Test the algorithm with a simple 2D example."""
    np.random.seed(42)

    # Create test data: 3 ellipsoids in 2D
    points = np.array([
        [0, 0],
        [2, 0],
        [1, 1.5]
    ])

    # Create precision matrices (A_i^{-1})
    A_inv = [
        np.array([[2, 0.5], [0.5, 1]]),  # Ellipse 1
        np.array([[1.5, -0.3], [-0.3, 2]]),  # Ellipse 2
        np.array([[3, 0], [0, 0.8]])  # Ellipse 3
    ]

    # Test with different epsilon values
    epsilons = [0.5, 1.0, 1.5, 2.0]

    for eps in epsilons:
        print(f"\n" + "=" * 50)
        print(f"Testing with ε = {eps}")
        print("=" * 50)

        optimizer = EllipsoidIntersectionOptimizer(points, A_inv, epsilon=eps)
        result = optimizer.minimize()

        print(f"\nResults for ε = {eps}:")
        print(f"  λ* = {result['lambda_optimal']}")
        print(f"  K(λ*) = {result['objective_value']:.6f}")
        print(f"  Centroid m(λ*) = {result['centroid']}")
        print(f"  Intersection exists: {result['intersection_exists']}")


if __name__ == "__main__":
    test_algorithm()