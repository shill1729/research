import numpy as np
import matplotlib.pyplot as plt

# Parameters
dt = 0.01  # Time step
T = 5  # Total time
N = int(T / dt)  # Number of steps

# Initialize arrays for A, B
theta_A = np.zeros(N)
theta_B = np.zeros(N)
theta_A[0] = np.random.uniform(0.1, 0.9)  # Initial value away from 0 and 1
theta_B[0] = np.random.uniform(0.1, 0.9)


def euler_maruyama(theta_0, dt, T, absorption_threshold=1e-3):
    """Simulates the stochastic process using the Euler-Maruyama method
       with absorption at 0 or 1.

    Args:
        theta_0 (float): Initial belief state.
        dt (float): Time step.
        T (float): Total simulation time.
        absorption_threshold (float): The threshold for absorption at 0 or 1.

    Returns:
        np.array: The process trajectory with absorption stopping.
    """
    N = int(T / dt)
    theta = np.zeros(N)
    theta[0] = theta_0

    for t in range(1, N):
        if theta[t - 1] <= absorption_threshold:
            theta[t:] = 0  # Pad with 0s after absorption
            break
        elif theta[t - 1] >= 1 - absorption_threshold:
            theta[t:] = 1  # Pad with 1s after absorption
            break

        # Compute drift and diffusion
        drift = -0.5 * (theta[t - 1] - 0.5) * dt
        diffusion = np.sqrt(theta[t - 1] * (1 - theta[t - 1])) * np.sqrt(dt) * np.random.randn()

        # Euler-Maruyama step
        theta[t] = theta[t - 1] + drift + diffusion

    return theta


# Parameters
dt = 0.001  # Time step
T = 5  # Total time

# Initial values
theta_A_0 = np.random.uniform(0.9, 0.91)
theta_B_0 = np.random.uniform(0.1, 0.11)

# Simulate processes
theta_A = euler_maruyama(theta_A_0, dt, T)
theta_B = euler_maruyama(theta_B_0, dt, T)

# Compute logical operations
theta_A_and_B = theta_A * theta_B
theta_A_or_B = theta_A + theta_B - theta_A_and_B
theta_A_implies_B = 1 - theta_A + theta_A * theta_B

# Plot results
time = np.linspace(0, T, len(theta_A))

plt.figure(figsize=(10, 6))
plt.plot(time, theta_A, label=r"$\theta^A$", linewidth=2)
plt.plot(time, theta_B, label=r"$\theta^B$", linewidth=2)
plt.plot(time, theta_A_and_B, label=r"$\theta^{A \wedge B}$", linestyle='dashed', linewidth=2)
plt.plot(time, theta_A_or_B, label=r"$\theta^{A \vee B}$", linestyle='dashed', linewidth=2)
plt.plot(time, theta_A_implies_B, label=r"$\theta^{A \Rightarrow B}$", linestyle='dotted', linewidth=2)

plt.xlabel("Time")
plt.ylabel("Belief")
plt.title("Stochastic Evolution of Logical Propositions with Absorption")
plt.legend()
plt.grid(True)
plt.show()

theta = np.linspace(0, 1, 1000)


def tau(theta):
    return np.pi ** 2 / 4 - np.arcsin(2 * theta - 1) ** 2


plt.plot(theta, tau(theta))
plt.xlabel("$\\theta$")
plt.ylabel("$\\tau(\\theta)$")
plt.title("$\\tau(\\theta)=E(T|\\theta_0=\\theta)$")
plt.show()

theta = np.linspace(0, 1, 1000)


# def h(theta):
#     return np.arctan2(4 * np.sqrt(-(theta - 1) * theta), 4 * theta - 2) / np.pi


def h(theta):
    return 0.5 - np.arcsin(2 * theta - 1) / np.pi


# plt.plot(theta, h(theta))
plt.plot(theta, h(theta))
plt.xlabel("$\\theta$")
plt.ylabel("$h(\\theta)$")
plt.title("$h(\\theta)=P(T_0 < T_1|\\theta_0=\\theta)$")
plt.show()
