import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from numpy.linalg import eig

def compute_correct_mvee(S, tol=1e-8, max_iter=1000):
    k, D = S.shape
    Q = np.column_stack((S, np.ones(k)))
    u = np.ones(k) / k

    for _ in range(max_iter):
        X = Q.T @ np.diag(u) @ Q
        try:
            invX = np.linalg.inv(X)
        except np.linalg.LinAlgError:
            break
        M_diag = np.diag(Q @ invX @ Q.T)
        j = np.argmax(M_diag)
        step_size = (M_diag[j] - D - 1) / ((D + 1) * (M_diag[j] - 1))
        new_u = (1 - step_size) * u
        new_u[j] += step_size
        if np.linalg.norm(new_u - u) < tol:
            u = new_u
            break
        u = new_u

    # second moment matrix
    c = S.T @ u
    X_centered = S - c.T
    M = X_centered.T @ np.diag(u) @ X_centered
    # shape matrix: A = D * M
    A = D * M
    return c, A

def plot_fixed_mvee(S, ax, title):
    try:
        c, A = compute_correct_mvee(S)
        eigvals, eigvecs = eig(A)
        order = np.argsort(eigvals)[::-1]
        eigvals = eigvals[order]
        eigvecs = eigvecs[:, order]

        theta = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * np.sqrt(eigvals)

        ax.scatter(S[:, 0], S[:, 1], color='black')
        ell = Ellipse(xy=c, width=width, height=height, angle=theta,
                      edgecolor='red', fc='none', lw=2)
        ax.add_patch(ell)
    except Exception:
        ax.scatter(S[:, 0], S[:, 1], color='black')
        ax.text(0.5, 0.5, "Singular (ill-posed)", ha='center', va='center', transform=ax.transAxes)
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_aspect('equal')
    ax.set_title(title)

np.random.seed(0)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

S2 = np.array([[0, 0], [2, 0]])
plot_fixed_mvee(S2, axes[0], "k=2 (ill-posed in $\\mathbb{R}^2$)")

S3 = np.array([[0, 0], [2, 0], [1, 2]])
plot_fixed_mvee(S3, axes[1], "k=3 (simplex)")

S4 = np.random.randn(4, 2)
plot_fixed_mvee(S4, axes[2], "k=4 (general full-rank)")

plt.tight_layout()
plt.show()
