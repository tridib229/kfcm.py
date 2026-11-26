
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# 1. LOAD DATASET
# =====================================================

file_path = Path(__file__).with_name("test_kfcm.data")

# Load whitespace-separated or comma-separated numeric data
try:
    X = np.loadtxt(file_path)
except ValueError:
    X = np.loadtxt(file_path, delimiter=",")

# Ensure X is a 2D array
if X.ndim == 1:
    X = X.reshape(1, -1)

# Check dataset
if X.shape[0] == 0 or X.shape[1] < 2:
    raise ValueError("Dataset must contain at least two numeric columns.")

if not np.isfinite(X).all():
    raise ValueError("Dataset contains missing or invalid values.")

n = X.shape[0]

print("Dataset loaded successfully.")
print("Number of data points:", n)
print("Number of features:", X.shape[1])

# =====================================================
# 2. PARAMETERS
# =====================================================

c = 2                   # Number of clusters
m = 1.5                 # Fuzzy weighting exponent
sigma2 = 0.5           # Gaussian kernel parameter
epsilon = 1e-6          # Convergence threshold
MAX_ITER = 100000

if m <= 1:
    raise ValueError("m must be greater than 1.")

if sigma2 <= 0:
    raise ValueError("sigma2 must be positive.")

# =====================================================
# 3. GAUSSIAN RBF KERNEL
# =====================================================

def gaussian_kernel(X, sigma2):

    squared_norms = np.sum(X ** 2, axis=1)

    squared_distances = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2 * X @ X.T
    )

    squared_distances = np.maximum(squared_distances, 0)

    K = np.exp(-squared_distances / sigma2)

    return K


K = gaussian_kernel(X, sigma2)

# =====================================================
# 4. INITIALIZE MEMBERSHIP MATRIX
# =====================================================

np.random.seed(42)

U = np.random.rand(n, c)

U = U / np.sum(U, axis=1, keepdims=True)

# =====================================================
# 5. KERNEL DISTANCE CALCULATION
# =====================================================

def calculate_kernel_distance(U, K, m):

    n, c = U.shape

    D = np.zeros((n, c))

    for i in range(c):

        weights = U[:, i] ** m

        weight_sum = np.sum(weights)

        if weight_sum <= 1e-12:
            raise ValueError("Cluster has zero membership weight.")

        # First term: K(x_k, x_k)
        term1 = np.diag(K)

        # Second term
        term2 = 2 * (K @ weights) / weight_sum

        # Third term
        term3 = (
            weights @ K @ weights
        ) / (weight_sum ** 2)

        # Kernel distance
        D[:, i] = term1 - term2 + term3

    # Avoid numerical negative distances
    D = np.maximum(D, 0)

    return D

# =====================================================
# 6. MEMBERSHIP MATRIX UPDATE
# =====================================================

def update_membership(D, m):

    n, c = D.shape

    U_new = np.zeros((n, c))

    power = 1.0 / (m - 1.0)

    for k in range(n):

        zero_clusters = np.where(D[k] <= 1e-12)[0]

        if len(zero_clusters) > 0:

            U_new[k, zero_clusters] = (
                1.0 / len(zero_clusters)
            )

        else:

            values = D[k] ** (-power)

            U_new[k] = values / np.sum(values)

    return U_new

# =====================================================
# 7. KFCM-K ITERATION
# =====================================================

print("\nStarting KFCM-K clustering...")

for iteration in range(MAX_ITER):

    U_old = U.copy()

    # Calculate kernel distances
    D = calculate_kernel_distance(U, K, m)

    # Update membership matrix
    U = update_membership(D, m)

    # Calculate convergence difference
    change = np.linalg.norm(U - U_old)

    print(
        f"Iteration {iteration + 1}: "
        f"Error = {change:.10f}"
    )

    # Check convergence
    if change < epsilon:

        print("\nConvergence reached.")

        break

else:

    print("\nMaximum iterations reached.")

# =====================================================
# 8. FINAL CLUSTER LABELS
# =====================================================

labels = np.argmax(U, axis=1)

# =====================================================
# 9. DISPLAY RESULTS
# =====================================================

print("\n===================================")
print("         KFCM-K RESULTS")
print("===================================")

print("\nNumber of data points:", n)

print("Number of clusters:", c)

print("Number of iterations:", iteration + 1)

print("\nFinal Membership Matrix:")

print(U)

print("\nFinal Cluster Labels:")

print(labels)

for i in range(c):

    print(
        f"Number of points in Cluster {i + 1}: "
        f"{np.sum(labels == i)}"
    )

# =====================================================
# 10. PLOT CLUSTERING RESULTS
# =====================================================

plt.figure(figsize=(8, 6))

for i in range(c):

    cluster_points = X[labels == i]

    plt.scatter(
        cluster_points[:, 0],
        cluster_points[:, 1],
        label=f"Cluster {i + 1}",
        s=35
    )

plt.xlabel("Feature 1")

plt.ylabel("Feature 2")

plt.title("KFCM-K Clustering Results")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.show()
