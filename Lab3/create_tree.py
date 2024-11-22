from sklearn.decomposition import TruncatedSVD
import numpy as np

# Example matrix (for instance, a term-document matrix)
X = np.random.rand(10, 5)

# Set the number of components (k) you want to retain
k = 2

# Apply Truncated SVD
svd = TruncatedSVD(n_components=k)
X_reduced = svd.fit_transform(X)

# U, D, V matrices
U = svd.transform(X)  # This is the left singular vectors
S = np.diag(svd.singular_values_)  # This is the diagonal matrix of singular values
V = svd.components_  # This is the right singular vectors (V.T)

print("Original matrix shape:", X.shape)
print("Transformed matrix (X_reduced) shape:", X_reduced.shape)

print("\nU matrix shape:", U.shape)
print("S matrix (diagonal singular values):", S.shape)
print("V matrix (right singular vectors):", V.shape)

print(X)
print(U@S@V)

target_trues = X.shape[0]*X.shape[1]
results = np.abs(U@S@V - X) < 1e-8
print(np.sum(results), target_trues)