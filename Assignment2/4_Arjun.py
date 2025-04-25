
import random
import math
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Error Handling
if len(sys.argv) != 8:
    sys.exit("Invalid Number of Parameters")

mu_1 = float(sys.argv[1])
mu_2 = float(sys.argv[2])
K_11 = float(sys.argv[3])
K_12 = float(sys.argv[4])
K_21 = float(sys.argv[5])
K_22 = float(sys.argv[6])
N = int(sys.argv[7])

mean = np.array([mu_1, mu_2])
K = np.array([[K_11, K_12], [K_21, K_22]])

if not np.allclose(K, K.T):
    sys.exit("Matrix should be symmetric")
if not np.all(np.linalg.eigvals(K) >= 0):
    sys.exit("Matrix should be Positive semidefinite")

print(mean)
print(K)

# Generate samples using scipy multivariate normal
X = stats.multivariate_normal.rvs(mean, K, N, random_state = 42)

# Spectral Decomposition
eigen_values, eigen_vectors = np.linalg.eigh(K)
D = np.diag(eigen_values)
U = eigen_vectors

A = U @ np.sqrt(D)
S = np.random.normal(0, 1, size=(2, N))
gaussian_vectors_custom = (A @ S).T + mean

# Set up the grid for the contour plot
x1 = np.arange(-mean[0] - 3 * np.sqrt(K[0, 0]), mean[0] + 3 * np.sqrt(K[0, 0]), 0.01)
x2 = np.arange(-mean[1] - 3 * np.sqrt(K[1, 1]), mean[1] + 3 * np.sqrt(K[1, 1]), 0.01)
X1, X2 = np.meshgrid(x1, x2)
Xpos = np.empty(X1.shape + (2,))
Xpos[:, :, 0] = X1
Xpos[:, :, 1] = X2
F = stats.multivariate_normal.pdf(Xpos, [0, 0], [[0.25, 0.3], [0.3, 1.0]])

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 7))

# Plot the contour plot and scatter plot for the custom Gaussian vectors
axes[0].contour(x1, x2, F, cmap = 'plasma')
axes[0].scatter(gaussian_vectors_custom[:, 0], gaussian_vectors_custom[:, 1], color='plum')
axes[0].set_title('Custom Gaussian Vectors')
axes[0].set_xlabel('X1')
axes[0].set_ylabel('X2')

# Plot the contour plot and scatter plot for the X from multivariate_normal.rvs
axes[1].contour(x1, x2, F, cmap = 'plasma')
axes[1].scatter(X[:, 0], X[:, 1], color='chartreuse')
axes[1].set_title('Multivariate Normal Samples')
axes[1].set_xlabel('X1')
axes[1].set_ylabel('X2')

# Show the plot
plt.tight_layout()
plt.show()
