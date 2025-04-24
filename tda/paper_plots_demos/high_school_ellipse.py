import numpy as np
import matplotlib.pyplot as plt

# Ellipse parameters
a = 5       # semi-major axis
b = 3       # semi-minor axis
h, k = 2, 1 # center of the ellipse

# Parametric equations
theta = np.linspace(0, 2 * np.pi, 400)
x = h + a * np.cos(theta)
y = k + b * np.sin(theta)

# Plot ellipse
fig, ax = plt.subplots()
ax.plot(x, y, label='Ellipse')
ax.set_aspect('equal')

# Axes and center
ax.plot(h, k, 'ko')  # center
ax.text(h + 0.2, k + 0.2, 'Center', fontsize=10)

# Semi-major axis
ax.plot([h - a, h + a], [k, k], 'r--', lw=1)
ax.text(h + a / 2, k + 0.2, 'Semi-major $a$', color='r', fontsize=10, ha='center')

# Semi-minor axis
ax.plot([h, h], [k - b, k + b], 'b--', lw=1)
ax.text(h + 0.2, k + b / 2, 'Semi-minor $b$', color='b', fontsize=10, va='center')

ax.set_title('Ellipse')
ax.grid(True)
plt.show()
