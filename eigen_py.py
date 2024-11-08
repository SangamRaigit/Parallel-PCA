import numpy as NP
from scipy import linalg as LA

# Input matrix
A = [[-1, 5, -9, 1], [9, 6, 0, 2], [2, -5, 1, 3]]
A = NP.array(A, dtype=float)

print("Original matrix A:\n", A)

# Perform SVD
u, s, vh = NP.linalg.svd(A, full_matrices=True)

# Display results
print("U matrix:\n", u)
print("Singular values (sigma):\n", s)
print("V_T matrix:\n", vh)

# Convert singular values into a diagonal matrix
sigma = NP.zeros((u.shape[0], vh.shape[0]))
NP.fill_diagonal(sigma, s)

# Check by reconstructing the matrix A using U, sigma, and V_T
reconstructed_A = NP.dot(u, NP.dot(sigma, vh))

print("Reconstructed matrix A:\n", reconstructed_A)
