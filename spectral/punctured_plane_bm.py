from sdes import SDE
import numpy as np
import matplotlib.pyplot as plt

x0 = np.array([0.5, 0.5])
tn = 2.5
ntime = 20000
npaths = 30

def mu(t, x):
    return np.zeros(2)

def sigma(t, x):
    r = np.linalg.vector_norm(x, axis=0, ord=2)
    return np.diag([r,r])


sde = SDE(mu, sigma)
ensemble = sde.sample_ensemble(x0, tn, ntime, npaths)

# Plot of ensemble
fig = plt.figure()
for i in range(npaths):
    plt.plot(ensemble[i, :, 0], ensemble[i, :, 1], alpha=0.5, c="blue")
plt.title("Riemannian BM in $\mathbb{C}\setminus \{0\}$")
plt.xlabel("$\Re(z)$")
plt.ylabel("$\Im(z)$")
plt.grid()
plt.show()

# Plot of standard BM(\C) versus this Riemannian one:
euclidean_sde = SDE(mu, lambda t, x: np.eye(2, 2))
planar_bm = euclidean_sde.solve(x0, tn, ntime, 0, 17, 2)
riemannian_bm = sde.solve(x0, tn, ntime, 0, 17, 2)

fig = plt.figure()
plt.plot(planar_bm[:, 0], planar_bm[:, 1], alpha=0.5, c="black", label="Euclidean BM")
plt.plot(riemannian_bm[:, 0], riemannian_bm[:, 1], alpha=0.5, c="blue", label="Riemannian BM")
plt.scatter(x0[0], x0[1], c="red")
plt.title("Riemannian BM in $\mathbb{C}\setminus \{0\}$ vs Standard BM")
plt.xlabel("$\Re(z)$")
plt.ylabel("$\Im(z)$")
plt.grid()
plt.legend()
plt.show()

