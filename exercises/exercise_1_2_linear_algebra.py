"""
Exercise 1.2: Linear Algebra for ML
Objective: Apply linear algebra concepts to real ML problems.

Tasks:
1. Implement matrix operations without using NumPy
2. Compute eigenvalues and eigenvectors
3. Apply SVD for dimensionality reduction
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# ============================================================
# Part 1: Matrix Operations Without NumPy
# ============================================================

def matrix_add(A, B):
    """Add two matrices."""
    if len(A) != len(B) or len(A[0]) != len(B[0]):
        raise ValueError("Matrices must have same dimensions")
    return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

def matrix_multiply(A, B):
    """Multiply two matrices."""
    if len(A[0]) != len(B):
        raise ValueError("Incompatible matrix dimensions")
    
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    
    return result

def matrix_transpose(A):
    """Transpose a matrix."""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

def matrix_scalar_multiply(A, scalar):
    """Multiply matrix by scalar."""
    return [[A[i][j] * scalar for j in range(len(A[0]))] for i in range(len(A))]

def dot_product(v1, v2):
    """Compute dot product of two vectors."""
    return sum(a * b for a, b in zip(v1, v2))

def vector_norm(v):
    """Compute L2 norm of a vector."""
    return sum(x**2 for x in v) ** 0.5

def test_matrix_operations():
    """Test the matrix operations."""
    print("Testing Matrix Operations (without NumPy)")
    print("=" * 50)
    
    A = [[1, 2, 3], [4, 5, 6]]
    B = [[7, 8, 9], [10, 11, 12]]
    
    print(f"Matrix A: {A}")
    print(f"Matrix B: {B}")
    
    C = matrix_add(A, B)
    print(f"A + B: {C}")
    
    D = matrix_scalar_multiply(A, 2)
    print(f"2 * A: {D}")
    
    E = matrix_transpose(A)
    print(f"A^T: {E}")
    
    # Matrix multiplication
    F = [[1, 2], [3, 4], [5, 6]]
    G = [[7, 8], [9, 10]]
    H = matrix_multiply(F, G)
    print(f"F @ G: {H}")

# ============================================================
# Part 2: Eigenvalues and Eigenvectors
# ============================================================

def power_iteration(A, num_iterations=100):
    """
    Power iteration method to find the dominant eigenvalue and eigenvector.
    
    Parameters:
    - A: square matrix (list of lists or numpy array)
    - num_iterations: number of iterations
    
    Returns:
    - eigenvalue: dominant eigenvalue
    - eigenvector: corresponding eigenvector
    """
    n = len(A)
    # Random initial vector
    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k)
    
    for _ in range(num_iterations):
        # Multiply by A
        b_k1 = np.dot(A, b_k)
        
        # Normalize
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
    
    # Compute eigenvalue using Rayleigh quotient
    eigenvalue = np.dot(b_k, np.dot(A, b_k)) / np.dot(b_k, b_k)
    
    return eigenvalue, b_k

def deflation(A, eigenvalue, eigenvector):
    """
    Deflate the matrix to find the next eigenvalue.
    
    A_new = A - eigenvalue * eigenvector * eigenvector^T
    """
    eigenvector = eigenvector.reshape(-1, 1)
    return A - eigenvalue * np.dot(eigenvector, eigenvector.T)

def compute_eigenvalues(A, num_eigenvalues=None):
    """
    Compute eigenvalues and eigenvectors using power iteration with deflation.
    
    Parameters:
    - A: square symmetric matrix
    - num_eigenvalues: number of eigenvalues to compute (default: all)
    
    Returns:
    - eigenvalues: list of eigenvalues
    - eigenvectors: list of eigenvectors
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    
    if num_eigenvalues is None:
        num_eigenvalues = n
    
    eigenvalues = []
    eigenvectors = []
    
    A_current = A.copy()
    
    for i in range(num_eigenvalues):
        eigenvalue, eigenvector = power_iteration(A_current)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
        
        # Deflate for next iteration
        if i < num_eigenvalues - 1:
            A_current = deflation(A_current, eigenvalue, eigenvector)
    
    return eigenvalues, eigenvectors

def test_eigenvalues():
    """Test eigenvalue computation."""
    print("\n\nEigenvalue Computation")
    print("=" * 50)
    
    # Create a symmetric matrix
    A = [[4, 1, 1],
         [1, 3, 1],
         [1, 1, 2]]
    
    print(f"Matrix A:\n{np.array(A)}")
    
    # Compute using our method
    eigenvalues, eigenvectors = compute_eigenvalues(A)
    
    print("\nComputed Eigenvalues:")
    for i, ev in enumerate(eigenvalues):
        print(f"  λ{i+1} = {ev:.4f}")
    
    # Compare with numpy
    np_eigenvalues, np_eigenvectors = np.linalg.eig(A)
    np_eigenvalues = sorted(np_eigenvalues, reverse=True)
    
    print("\nNumPy Eigenvalues (for comparison):")
    for i, ev in enumerate(np_eigenvalues):
        print(f"  λ{i+1} = {ev:.4f}")

# ============================================================
# Part 3: SVD for Dimensionality Reduction
# ============================================================

def svd_reduction(data, n_components):
    """
    Apply SVD for dimensionality reduction.
    
    Parameters:
    - data: numpy array of shape (n_samples, n_features)
    - n_components: number of components to keep
    
    Returns:
    - reduced_data: data projected onto principal components
    - components: principal components
    - explained_variance: variance explained by each component
    """
    # Center the data
    mean = np.mean(data, axis=0)
    data_centered = data - mean
    
    # Compute SVD
    U, S, Vt = np.linalg.svd(data_centered, full_matrices=False)
    
    # Select top n_components
    U_reduced = U[:, :n_components]
    S_reduced = S[:n_components]
    Vt_reduced = Vt[:n_components, :]
    
    # Project data
    reduced_data = U_reduced * S_reduced
    
    # Compute explained variance
    explained_variance = (S_reduced ** 2) / (len(data) - 1)
    total_variance = np.sum(S ** 2) / (len(data) - 1)
    explained_variance_ratio = explained_variance / total_variance
    
    return reduced_data, Vt_reduced, explained_variance_ratio

def visualize_svd_reduction():
    """Visualize SVD-based dimensionality reduction on Iris dataset."""
    print("\n\nSVD for Dimensionality Reduction")
    print("=" * 50)
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Original data shape: {X_scaled.shape}")
    print(f"Features: {feature_names}")
    
    # Apply SVD reduction to 2 components
    reduced_data, components, explained_variance_ratio = svd_reduction(X_scaled, 2)
    
    print(f"\nReduced data shape: {reduced_data.shape}")
    print(f"Explained variance ratio:")
    for i, var in enumerate(explained_variance_ratio):
        print(f"  Component {i+1}: {var:.4f} ({var*100:.2f}%)")
    print(f"Total variance explained: {sum(explained_variance_ratio)*100:.2f}%")
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Original data (first two features)
    for i, (color, label) in enumerate(zip(['red', 'green', 'blue'], target_names)):
        mask = y == i
        ax1.scatter(X[mask, 0], X[mask, 1], c=color, label=label, alpha=0.7, s=50)
    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])
    ax1.set_title('Original Data (First Two Features)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Reduced data
    for i, (color, label) in enumerate(zip(['red', 'green', 'blue'], target_names)):
        mask = y == i
        ax2.scatter(reduced_data[mask, 0], reduced_data[mask, 1], c=color, label=label, alpha=0.7, s=50)
    ax2.set_xlabel('Principal Component 1')
    ax2.set_ylabel('Principal Component 2')
    ax2.set_title('SVD-Reduced Data (2 Components)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svd_reduction.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Visualize reconstruction error
    n_components_range = range(1, 5)
    reconstruction_errors = []
    
    for n_comp in n_components_range:
        reduced, _, _ = svd_reduction(X_scaled, n_comp)
        # Reconstruct
        U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
        reconstructed = np.dot(U[:, :n_comp] * S[:n_comp], Vt[:n_comp, :])
        error = np.mean((X_scaled - reconstructed) ** 2)
        reconstruction_errors.append(error)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(n_components_range, reconstruction_errors, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Reconstruction Error (MSE)')
    ax.set_title('Reconstruction Error vs Number of Components')
    ax.grid(True, alpha=0.3)
    ax.set_xticks(n_components_range)
    plt.tight_layout()
    plt.savefig('reconstruction_error.png', dpi=150, bbox_inches='tight')
    plt.show()

def pca_from_scratch():
    """Implement PCA from scratch using SVD."""
    print("\n\nPCA from Scratch using SVD")
    print("=" * 50)
    
    # Load Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply SVD
    U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
    
    # Transform to 2D
    X_pca = U[:, :2] * S[:2]
    
    # Visualize
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for i, (color, label) in enumerate(zip(['red', 'green', 'blue'], iris.target_names)):
        mask = y == i
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=color, label=label, alpha=0.7, s=50)
    
    ax.set_xlabel(f'PC1 ({S[0]**2/np.sum(S**2)*100:.1f}% variance)')
    ax.set_ylabel(f'PC2 ({S[1]**2/np.sum(S**2)*100:.1f}% variance)')
    ax.set_title('PCA of Iris Dataset (from Scratch)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_from_scratch.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Exercise 1.2: Linear Algebra for ML")
    print("=" * 60)
    
    # Part 1: Matrix Operations
    test_matrix_operations()
    
    # Part 2: Eigenvalues
    test_eigenvalues()
    
    # Part 3: SVD
    visualize_svd_reduction()
    pca_from_scratch()
    
    print("\n\nExercise Complete!")
    print("Write a brief report explaining:")
    print("1. How matrix operations form the basis of ML algorithms")
    print("2. The significance of eigenvalues in understanding data")
    print("3. How SVD enables dimensionality reduction")