"""
    This module implements the function $K(\\lambda)$ from our paper.

"""
import numpy as np
import time


def get_A_operations(A_list, xs):
    """

    :param A_list: the list of covariance matrices evaluated at x_i
    :param xs: the tuple of centers x_i
    :return:
    """
    # Precompute each A_i^{-1} and x_i^T A_i^{-1} x_i.
    k, _ = xs.shape
    A_inv_list = []
    x_Ainv_x = np.zeros(k)
    for i in range(k):
        A_inv = np.linalg.inv(A_list[i])
        A_inv_list.append(A_inv)
        x_Ainv_x[i] = xs[i].T @ A_inv @ xs[i]
    # Stack the inverses into a single array for vectorized operations.
    A_inv_array = np.stack(A_inv_list, axis=0)
    return A_inv_array, x_Ainv_x


def compute_K(lmbd, eps, xs, A_inv_array, x_Ainv_x):
    """
    Computes K(λ) = eps^2 - C(λ), where
       A_lambda_inv = sum_i λ_i A_inv_array[i]
       S = sum_i λ_i (A_inv_array[i] @ xs[i])
       m(λ) is obtained by solving A_lambda_inv @ m = S,
       and C(λ) = (sum_i λ_i x_Ainv_x[i]) - m(λ)^T S.

    :param lmbd: convex combination coefficients
    :param eps: the radius/persistence parameter
    :param xs: the centers, the pair, the tuple, triple
    :param A_inv_array: the tensor of inverse A inverse.
    :param x_Ainv_x: the quadratic form at the centers.
    :return:
    """
    A_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))
    S = np.sum(lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)
    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        print("Non PD!")
        return 1e10  # Penalty for singularity
    quad_term = m_lambda.dot(S)  # equals m_lambda^T A_lambda_inv m_lambda
    sum_term = np.dot(lmbd, x_Ainv_x)
    return eps ** 2 - (sum_term - quad_term)


def compute_K_gradient(lmbd, xs, A_inv_array):
    """
    Computes the gradient ∇K(λ) using the simplified formula:
      ∂K/∂λ_j = -(x_j - m(λ))^T A_j^{-1} (x_j - m(λ))
    where m(λ) is computed from the weighted combination of A_inv_array.

    :param lmbd:
    :param xs:
    :param A_inv_array:
    :return:
    """
    # Compute aggregated inverse and weighted sum
    A_lambda_inv = np.tensordot(lmbd, A_inv_array, axes=([0], [0]))
    S = np.sum(lmbd[:, None] * np.einsum('ijk,ik->ij', A_inv_array, xs), axis=0)
    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        return np.zeros_like(lmbd)

    # Compute (x_j - m(λ)) for each j
    diff = xs - m_lambda  # broadcasting m_lambda to each row of xs
    # For each j, compute (x_j - m(λ))^T A_inv_array[j] (x_j - m(λ)):
    grad = -np.einsum('ij,ijk,ik->i', diff, A_inv_array, diff)
    return grad


def compute_K_hessian(lmbd, xs, A_inv_array):
    """
    Naively computes the Hessian of K(λ) with entries:
      H[i,j] = 2 * (m(λ)-x_i)^T A_inv_array[i] B A_inv_array[j] (m(λ)-x_j)
    where:
      - A_lambda_inv = sum_i λ_i A_inv_array[i],
      - S = sum_i λ_i (A_inv_array[i] @ xs[i]),
      - m(λ) is obtained by solving A_lambda_inv @ m = S, and
      - B = inv(A_lambda_inv).

    This implementation uses explicit Python loops for clarity.

    Parameters:
      lmbd: 1D NumPy array of convex coefficients (length k).
      xs: NumPy array of centers with shape (k, d).
      A_inv_array: NumPy array of shape (k, d, d) containing the inverses of A(x_i).

    Returns:
      Hessian: NumPy array of shape (k, k) with the Hessian entries.
    """
    k, d, _ = A_inv_array.shape

    # Compute A_lambda_inv = sum_i λ_i A_inv_array[i]
    A_lambda_inv = np.zeros((d, d))
    for i in range(k):
        A_lambda_inv += lmbd[i] * A_inv_array[i]

    # Compute S = sum_i λ_i * (A_inv_array[i] @ xs[i])
    S = np.zeros(d)
    for i in range(k):
        S += lmbd[i] * (A_inv_array[i] @ xs[i])

    # Solve for m(λ)
    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        return np.zeros((k, k))

    # Compute B = inv(A_lambda_inv)
    B = np.linalg.inv(A_lambda_inv)

    # Compute the Hessian using nested loops
    H = np.zeros((k, k))
    for i in range(k):
        diff_i = xs[i] - m_lambda
        u_i = A_inv_array[i] @ diff_i
        for j in range(k):
            diff_j = xs[j] - m_lambda
            u_j = A_inv_array[j] @ diff_j
            H[i, j] = 2 * (u_i.T @ (B @ u_j))
    return H


def get_A_operations_fast(A_array, xs):
    """
    Compute A_inv_array and x_Ainv_x for a stack of covariance matrices.

    Parameters:
      A_array: NumPy array of shape (k, d, d), where each A_array[i] is A(x_i).
      xs: NumPy array of shape (k, d), the centers.

    Returns:
      A_inv_array: Array of shape (k, d, d) with each inverse.
      x_Ainv_x: Array of shape (k,) with each quadratic form x_i^T @ A_inv_array[i] @ x_i.
    """
    # Vectorized inversion (requires numpy>=1.8 for batched inverses)
    A_inv_array = np.linalg.inv(A_array)

    # Compute A_inv_array @ xs for each i
    Ax = np.matmul(A_inv_array, xs[..., None]).squeeze(-1)  # shape (k, d)
    # Compute quadratic forms: for each i, x_i^T A_inv_array[i] x_i
    x_Ainv_x = np.sum(xs * Ax, axis=1)

    return A_inv_array, x_Ainv_x


def compute_K_fast(lmbd, eps, xs, A_inv_array, x_Ainv_x):
    """
    Computes K(λ) = eps^2 - C(λ), where
      A_lambda_inv = sum_i λ_i A_inv_array[i],
      S = sum_i λ_i (A_inv_array[i] @ xs[i]),
      m(λ) is obtained by solving A_lambda_inv @ m = S,
      and C(λ) = (sum_i λ_i x_Ainv_x[i]) - m(λ)^T S.

    Parameters:
      lmbd: 1D NumPy array of convex coefficients (length k)
      eps: scalar (the radius parameter)
      xs: centers, shape (k, d)
      A_inv_array: precomputed inverses, shape (k, d, d)
      x_Ainv_x: precomputed quadratic forms, shape (k,)

    Returns:
      The scalar value K(λ).
    """
    # Compute the weighted sum of inverse matrices.
    A_lambda_inv = (lmbd[:, None, None] * A_inv_array).sum(axis=0)  # shape (d, d)

    # Compute S = sum_i λ_i * (A_inv_array[i] @ xs[i])
    Ax = np.matmul(A_inv_array, xs[..., None]).squeeze(-1)  # shape (k, d)
    S = (lmbd[:, None] * Ax).sum(axis=0)  # shape (d,)

    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        print("Non PD!")
        return 1e10  # Penalty for singularity

    quad_term = m_lambda.dot(S)  # equals m(λ)^T S
    sum_term = lmbd.dot(x_Ainv_x)

    return eps ** 2 - (sum_term - quad_term)


def compute_K_gradient_fast(lmbd, xs, A_inv_array):
    """
    Computes the gradient ∇K(λ) using the simplified formula:
      ∂K/∂λ_j = - (x_j - m(λ))^T A_j^{-1} (x_j - m(λ))
    where m(λ) is computed from the weighted combination of A_inv_array.

    Parameters:
      lmbd: 1D NumPy array of convex coefficients (length k)
      xs: centers, shape (k, d)
      A_inv_array: precomputed inverses, shape (k, d, d)

    Returns:
      Gradient as a 1D NumPy array of length k.
    """
    # Compute A_lambda_inv and S as before
    A_lambda_inv = (lmbd[:, None, None] * A_inv_array).sum(axis=0)
    Ax = np.matmul(A_inv_array, xs[..., None]).squeeze(-1)
    S = (lmbd[:, None] * Ax).sum(axis=0)

    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        return np.zeros_like(lmbd)

    # Compute difference vectors: (x_j - m(λ)) for each j
    diff = xs - m_lambda  # shape (k, d)

    # For each j, compute (x_j - m(λ))^T A_inv_array[j] (x_j - m(λ))
    v = np.matmul(A_inv_array, diff[..., None]).squeeze(-1)  # shape (k, d)
    grad = -np.sum(diff * v, axis=1)

    return grad


def compute_K_hessian_fast(lmbd, xs, A_inv_array):
    """
    Computes the Hessian of K(λ) with entries:
      H[i,j] = 2 * (m(λ)-x_i)^T A_inv_array[i] B A_inv_array[j] (m(λ)-x_j),
    where B = inv(sum_i λ_i A_inv_array[i]) (i.e. B(λ)).

    Parameters:
      lmbd: 1D NumPy array of convex coefficients (length k)
      xs: centers, shape (k, d)
      A_inv_array: precomputed inverses, shape (k, d, d)

    Returns:
      Hessian: a (k, k) NumPy array.
    """
    # Compute A_lambda_inv and S as in compute_K.
    A_lambda_inv = (lmbd[:, None, None] * A_inv_array).sum(axis=0)
    Ax = np.matmul(A_inv_array, xs[..., None]).squeeze(-1)
    S = (lmbd[:, None] * Ax).sum(axis=0)

    try:
        m_lambda = np.linalg.solve(A_lambda_inv, S)
    except np.linalg.LinAlgError:
        return np.zeros((lmbd.shape[0], lmbd.shape[0]))

    # Compute B = inv(A_lambda_inv)
    B = np.linalg.inv(A_lambda_inv)

    # Compute the difference vectors
    diff = xs - m_lambda  # shape (k, d)
    # Compute u[i] = A_inv_array[i] @ diff[i]
    u = np.matmul(A_inv_array, diff[..., None]).squeeze(-1)  # shape (k, d)

    # The Hessian entry H[i,j] = 2 * u[i]^T B u[j]
    # Vectorize by computing the matrix product:
    Hessian = 2 * (u @ (B @ u.T))

    return Hessian


def assess_runtime(k, d, A_list, A_array):
    # Generate random 2D centers
    xs = np.random.randn(k, d)

    # Generate a random probability vector λ
    lmbd = np.random.rand(k)
    lmbd = lmbd / lmbd.sum()
    eps = 1.0

    # --- Original Implementation ---
    A_inv, x_Ainv_x = get_A_operations(A_list, xs)
    K_val = compute_K(lmbd, eps, xs, A_inv, x_Ainv_x)
    grad_val = compute_K_gradient(lmbd, xs, A_inv)
    hess_val = compute_K_hessian(lmbd, xs, A_inv)

    # --- Fast, Vectorized Implementation ---
    A_inv_fast, x_Ainv_x_fast = get_A_operations_fast(A_array, xs)
    K_val_fast = compute_K_fast(lmbd, eps, xs, A_inv_fast, x_Ainv_x_fast)
    grad_val_fast = compute_K_gradient_fast(lmbd, xs, A_inv_fast)
    Hessian_fast = compute_K_hessian_fast(lmbd, xs, A_inv_fast)

    # Output the results and check closeness.
    print("K (original):", K_val)
    print("K (fast):", K_val_fast)
    print("Gradient (original):", grad_val)
    print("Gradient (fast):", grad_val_fast)
    print("Hessian (original):", hess_val)
    print("Hessian (fast):\n", Hessian_fast)
    print("K values close?", np.allclose(K_val, K_val_fast))
    print("Gradient values close?", np.allclose(grad_val, grad_val_fast))
    print("Hessian values close?", np.allclose(hess_val, Hessian_fast))

    # --- Timing tests ---
    niter = 100

    # Time get_A_operations (original)
    start = time.time()
    for _ in range(niter):
        _ = get_A_operations(A_list, xs)
    t_orig_get = time.time() - start

    # Time get_A_operations_fast
    start = time.time()
    for _ in range(niter):
        _ = get_A_operations_fast(A_array, xs)
    t_fast_get = time.time() - start

    # Time compute_K (original)
    start = time.time()
    for _ in range(niter):
        _ = compute_K(lmbd, eps, xs, A_inv, x_Ainv_x)
    t_orig_K = time.time() - start

    # Time compute_K_fast
    start = time.time()
    for _ in range(niter):
        _ = compute_K_fast(lmbd, eps, xs, A_inv_fast, x_Ainv_x_fast)
    t_fast_K = time.time() - start

    # Time compute_K_gradient (original)
    start = time.time()
    for _ in range(niter):
        _ = compute_K_gradient(lmbd, xs, A_inv)
    t_orig_grad = time.time() - start

    # Time compute_K_gradient_fast
    start = time.time()
    for _ in range(niter):
        _ = compute_K_gradient_fast(lmbd, xs, A_inv_fast)
    t_fast_grad = time.time() - start

    # Time compute_K_hessian (original)
    start = time.time()
    for _ in range(niter):
        _ = compute_K_hessian(lmbd, xs, A_inv)
    t_orig_grad = time.time() - start

    # Time compute_K_gradient_fast
    start = time.time()
    for _ in range(niter):
        _ = compute_K_hessian_fast(lmbd, xs, A_inv_fast)
    t_fast_hess = time.time() - start

    print("\nTiming over {} iterations:".format(niter))
    print("get_A_operations  - original: {:.6f}s, fast: {:.6f}s".format(t_orig_get, t_fast_get))
    print("compute_K         - original: {:.6f}s, fast: {:.6f}s".format(t_orig_K, t_fast_K))
    print("compute_K_gradient- original: {:.6f}s, fast: {:.6f}s".format(t_orig_grad, t_fast_grad))
    print("compute_K_hessian- original: {:.6f}s, fast: {:.6f}s".format(t_orig_grad, t_fast_hess))


def generate_random_A(k, d):
    # Create k random 2x2 PD matrices: A = M.T @ M + I ensures positive definiteness.
    A_list = []
    A_array = np.empty((k, d, d))
    for i in range(k):
        M = np.random.randn(d, d)
        A = M.T @ M + np.eye(d)
        A_list.append(A)
        A_array[i] = A
    return A_list, A_array


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    # k = 3  # number of centers
    # d = 2  # dimensionality

    def runtime_wrapper(k, d):
        print("\nTesting "+str(k)+"-many centers in "+str(d)+" dimensions")
        A_list, A_array = generate_random_A(k, d)
        return assess_runtime(k, d, A_list, A_array)

    # Pairs
    runtime_wrapper(k=2, d=2)
    runtime_wrapper(k=2, d=3)
    runtime_wrapper(k=2, d=50)
    runtime_wrapper(k=2, d=100)
    runtime_wrapper(k=2, d=500)
    # Triples
    runtime_wrapper(k=3, d=2)
    runtime_wrapper(k=3, d=3)
    runtime_wrapper(k=3, d=50)
    runtime_wrapper(k=3, d=100)
    runtime_wrapper(k=3, d=500)
    # 4-tuples
    runtime_wrapper(k=4, d=2)
    runtime_wrapper(k=4, d=3)
    runtime_wrapper(k=4, d=50)
    runtime_wrapper(k=4, d=100)
    runtime_wrapper(k=4, d=500)
