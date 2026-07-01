"""
Exercise 2.2: Clustering Analysis
Objective: Apply unsupervised learning techniques.

Requirements:
- Implement K-Means clustering from scratch
- Apply hierarchical clustering
- Use PCA for visualization
- Analyze cluster characteristics
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================
# Part 1: K-Means Clustering from Scratch
# ============================================================

class KMeansScratch:
    """
    K-Means clustering algorithm implemented from scratch.
    
    K-Means is an iterative algorithm that:
    1. Initializes K centroids randomly
    2. Assigns each data point to the nearest centroid
    3. Updates centroids as the mean of assigned points
    4. Repeats steps 2-3 until convergence
    """
    
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        """
        Initialize K-Means.
        
        Parameters:
        -----------
        n_clusters : int
            Number of clusters (K)
        max_iters : int
            Maximum number of iterations
        random_state : int
            Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centroids = None
        self.labels = None
        self.inertia_history = []
        
    def _euclidean_distance(self, X, centroids):
        """
        Compute Euclidean distance between each point and each centroid.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
        centroids : numpy array of shape (n_clusters, n_features)
        
        Returns:
        --------
        distances : numpy array of shape (n_samples, n_clusters)
        """
        # Using broadcasting for efficient computation
        # X shape: (n_samples, 1, n_features)
        # centroids shape: (1, n_clusters, n_features)
        distances = np.sqrt(((X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2))
        return distances
    
    def _initialize_centroids(self, X):
        """
        Initialize centroids using K-Means++ method for better initialization.
        
        K-Means++ selects centroids that are far apart from each other,
        which leads to better convergence.
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]
        
        # Choose first centroid randomly
        first_idx = np.random.randint(0, n_samples)
        centroids = [X[first_idx]]
        
        # Choose remaining centroids with probability proportional to distance squared
        for _ in range(1, self.n_clusters):
            # Compute distances to nearest centroid
            distances = self._euclidean_distance(X, np.array(centroids))
            min_distances = distances.min(axis=1)
            
            # Square distances for probability
            squared_distances = min_distances ** 2
            
            # Compute probabilities
            probabilities = squared_distances / squared_distances.sum()
            
            # Choose next centroid
            next_idx = np.random.choice(n_samples, p=probabilities)
            centroids.append(X[next_idx])
        
        return np.array(centroids)
    
    def fit(self, X):
        """
        Fit K-Means to the data.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
        
        Returns:
        --------
        self
        """
        # Initialize centroids
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            # Step 1: Assign points to nearest centroid
            distances = self._euclidean_distance(X, self.centroids)
            self.labels = distances.argmin(axis=1)
            
            # Step 2: Update centroids
            new_centroids = np.zeros_like(self.centroids)
            for k in range(self.n_clusters):
                mask = self.labels == k
                if mask.sum() > 0:
                    new_centroids[k] = X[mask].mean(axis=0)
                else:
                    # If cluster is empty, reinitialize randomly
                    new_centroids[k] = X[np.random.randint(0, X.shape[0])]
            
            # Compute inertia (within-cluster sum of squares)
            inertia = 0
            for k in range(self.n_clusters):
                mask = self.labels == k
                if mask.sum() > 0:
                    inertia += ((X[mask] - self.centroids[k]) ** 2).sum()
            self.inertia_history.append(inertia)
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids, rtol=1e-6):
                print(f"  Converged at iteration {iteration + 1}")
                self.centroids = new_centroids
                break
            
            self.centroids = new_centroids
        
        return self
    
    def predict(self, X):
        """
        Predict cluster labels for X.
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
        
        Returns:
        --------
        labels : numpy array of shape (n_samples,)
        """
        distances = self._euclidean_distance(X, self.centroids)
        return distances.argmin(axis=1)
    
    def inertia(self, X):
        """Compute within-cluster sum of squares."""
        inertia = 0
        for k in range(self.n_clusters):
            mask = self.labels == k
            if mask.sum() > 0:
                inertia += ((X[mask] - self.centroids[k]) ** 2).sum()
        return inertia


def kmeans_from_scratch_demo():
    """Demonstrate K-Means clustering implemented from scratch."""
    print("=" * 60)
    print("K-MEANS CLUSTERING FROM SCRATCH")
    print("=" * 60)
    
    # Generate sample data
    print("\nGenerating sample data with 4 clusters...")
    X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Data shape: {X_scaled.shape}")
    print(f"True number of clusters: {len(np.unique(y_true))}")
    
    # Visualize original data
    plt.figure(figsize=(8, 6))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=50)
    plt.title('Original Data (True Labels)')
    plt.xlabel('Feature 1 (standardized)')
    plt.ylabel('Feature 2 (standardized)')
    plt.colorbar(label='True Cluster')
    plt.tight_layout()
    plt.savefig('kmeans_original_data.png', dpi=150)
    plt.show()
    
    # Fit K-Means from scratch
    print("\nFitting K-Means (from scratch) with K=4...")
    kmeans = KMeansScratch(n_clusters=4, max_iters=100, random_state=42)
    kmeans.fit(X_scaled)
    
    y_pred = kmeans.labels
    
    # Visualize results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Cluster assignments
    axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_pred, cmap='viridis', alpha=0.6, s=50)
    axes[0].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
    axes[0].set_title('K-Means Clustering Results')
    axes[0].set_xlabel('Feature 1')
    axes[0].set_ylabel('Feature 2')
    axes[0].legend()
    
    # Inertia over iterations
    axes[1].plot(kmeans.inertia_history, 'b-', linewidth=2)
    axes[1].set_title('Inertia Over Iterations')
    axes[1].set_xlabel('Iteration')
    axes[1].set_ylabel('Inertia (WCSS)')
    axes[1].grid(True, alpha=0.3)
    
    # Comparison with true labels
    axes[2].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=50)
    axes[2].set_title('True Labels (for comparison)')
    axes[2].set_xlabel('Feature 1')
    axes[2].set_ylabel('Feature 2')
    
    plt.tight_layout()
    plt.savefig('kmeans_results.png', dpi=150)
    plt.show()
    
    return X_scaled, y_true


def find_optimal_k(X, max_k=10):
    """
    Find optimal number of clusters using Elbow Method and Silhouette Analysis.
    
    Parameters:
    -----------
    X : numpy array of shape (n_samples, n_features)
    max_k : int
        Maximum number of clusters to try
    """
    print("\n" + "=" * 60)
    print("FINDING OPTIMAL K")
    print("=" * 60)
    
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)
    
    for k in k_range:
        print(f"\nTesting K={k}...")
        kmeans = KMeansScratch(n_clusters=k, max_iters=100, random_state=42)
        kmeans.fit(X)
        
        inertias.append(kmeans.inertia(X))
        
        # Compute silhouette score
        sil_score = silhouette_score(X, kmeans.labels)
        silhouette_scores.append(sil_score)
        print(f"  Inertia: {kmeans.inertia(X):.2f}")
        print(f"  Silhouette Score: {sil_score:.4f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Elbow plot
    ax1.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    ax1.set_title('Elbow Method')
    ax1.set_xlabel('Number of Clusters (K)')
    ax1.set_ylabel('Inertia (WCSS)')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette plot
    ax2.plot(k_range, silhouette_scores, 'ro-', linewidth=2, markersize=8)
    ax2.set_title('Silhouette Analysis')
    ax2.set_xlabel('Number of Clusters (K)')
    ax2.set_ylabel('Silhouette Score')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimal_k_analysis.png', dpi=150)
    plt.show()
    
    # Recommend optimal K
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    print(f"\nOptimal K (Silhouette): {optimal_k_silhouette}")
    
    return inertias, silhouette_scores


# ============================================================
# Part 2: Hierarchical Clustering
# ============================================================

def hierarchical_clustering_demo(X):
    """
    Demonstrate hierarchical clustering with dendrogram visualization.
    
    Parameters:
    -----------
    X : numpy array of shape (n_samples, n_features)
    """
    print("\n" + "=" * 60)
    print("HIERARCHICAL CLUSTERING")
    print("=" * 60)
    
    # Compute linkage matrix using different methods
    methods = ['ward', 'complete', 'average', 'single']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, method in enumerate(methods):
        print(f"\nComputing hierarchical clustering with {method} linkage...")
        
        # Compute linkage matrix
        Z = linkage(X, method=method)
        
        # Plot dendrogram
        plt.sca(axes[idx])
        dendrogram(Z, truncate_mode='level', p=5, color_threshold=0.7 * Z[-1, 2])
        axes[idx].set_title(f'{method.capitalize()} Linkage')
        axes[idx].set_xlabel('Sample Index')
        axes[idx].set_ylabel('Distance')
    
    plt.tight_layout()
    plt.savefig('dendrograms.png', dpi=150)
    plt.show()
    
    # Get cluster labels by cutting the dendrogram
    print("\nCutting dendrogram to get 4 clusters...")
    Z_ward = linkage(X, method='ward')
    labels_hierarchical = fcluster(Z_ward, t=4, criterion='maxclust')
    
    # Visualize hierarchical clustering results
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=labels_hierarchical, cmap='viridis', alpha=0.6, s=50)
    plt.title('Hierarchical Clustering Results (Ward)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    plt.tight_layout()
    plt.savefig('hierarchical_results.png', dpi=150)
    plt.show()
    
    return labels_hierarchical


# ============================================================
# Part 3: PCA for Visualization
# ============================================================

def pca_visualization_demo(X, y_true=None):
    """
    Use PCA for dimensionality reduction and visualization.
    
    Parameters:
    -----------
    X : numpy array of shape (n_samples, n_features)
    y_true : numpy array of shape (n_samples,), optional
        True labels for coloring the plot
    """
    print("\n" + "=" * 60)
    print("PCA FOR CLUSTER VISUALIZATION")
    print("=" * 60)
    
    # Apply PCA
    print("\nApplying PCA for dimensionality reduction...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.4f}")
    
    # Visualize PCA results
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # PCA with true labels
    if y_true is not None:
        axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=50)
        axes[0].set_title('PCA: True Labels')
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    
    # K-Means on original data
    kmeans = KMeansScratch(n_clusters=4, random_state=42)
    kmeans.fit(X)
    axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans.labels, cmap='viridis', alpha=0.6, s=50)
    axes[1].set_title('K-Means Clusters (Original Data)')
    axes[1].set_xlabel(f'PC1')
    axes[1].set_ylabel(f'PC2')
    
    # K-Means on PCA-transformed data
    kmeans_pca = KMeansScratch(n_clusters=4, random_state=42)
    kmeans_pca.fit(X_pca)
    axes[2].scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_pca.labels, cmap='viridis', alpha=0.6, s=50)
    axes[2].set_title('K-Means on PCA-Transformed Data')
    axes[2].set_xlabel(f'PC1')
    axes[2].set_ylabel(f'PC2')
    
    plt.tight_layout()
    plt.savefig('pca_visualization.png', dpi=150)
    plt.show()
    
    return X_pca, pca


# ============================================================
# Part 4: Cluster Analysis and Evaluation
# ============================================================

def cluster_analysis(X, labels_dict):
    """
    Analyze and compare clustering results.
    
    Parameters:
    -----------
    X : numpy array of shape (n_samples, n_features)
    labels_dict : dict
        Dictionary containing different clustering labels
    """
    print("\n" + "=" * 60)
    print("CLUSTER ANALYSIS AND EVALUATION")
    print("=" * 60)
    
    results = []
    
    for name, labels in labels_dict.items():
        print(f"\nEvaluating: {name}")
        
        # Compute clustering metrics
        sil_score = silhouette_score(X, labels)
        db_score = davies_bouldin_score(X, labels)
        ch_score = calinski_harabasz_score(X, labels)
        
        print(f"  Silhouette Score: {sil_score:.4f} (higher is better)")
        print(f"  Davies-Bouldin Score: {db_score:.4f} (lower is better)")
        print(f"  Calinski-Harabasz Score: {ch_score:.4f} (higher is better)")
        
        # Cluster sizes
        unique, counts = np.unique(labels, return_counts=True)
        print(f"  Cluster sizes: {dict(zip(unique, counts))}")
        
        results.append({
            'method': name,
            'silhouette': sil_score,
            'davies_bouldin': db_score,
            'calinski_harabasz': ch_score
        })
    
    # Compare results
    results_df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("COMPARISON OF CLUSTERING METHODS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    metrics = ['silhouette', 'davies_bouldin', 'calinski_harabasz']
    titles = ['Silhouette Score', 'Davies-Bouldin Score', 'Calinski-Harabasz Score']
    colors = ['blue', 'green', 'red']
    
    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        ax.bar(results_df['method'], results_df[metric], color=color, alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('cluster_comparison.png', dpi=150)
    plt.show()
    
    return results_df


# ============================================================
# Part 5: Real Dataset Application (Iris Dataset)
# ============================================================

def iris_clustering_application():
    """Apply clustering techniques to the Iris dataset."""
    print("\n" + "=" * 60)
    print("REAL DATASET APPLICATION: IRIS")
    print("=" * 60)
    
    # Load Iris dataset
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    
    print(f"\nDataset shape: {X_iris.shape}")
    print(f"Features: {iris.feature_names}")
    print(f"True classes: {np.unique(y_iris)}")
    print(f"Class distribution: {np.bincount(y_iris)}")
    
    # Standardize
    scaler = StandardScaler()
    X_iris_scaled = scaler.fit_transform(X_iris)
    
    # Apply K-Means
    print("\nApplying K-Means clustering...")
    kmeans_iris = KMeansScratch(n_clusters=3, random_state=42)
    kmeans_iris.fit(X_iris_scaled)
    y_kmeans = kmeans_iris.labels
    
    # Apply hierarchical clustering
    print("Applying hierarchical clustering...")
    Z_iris = linkage(X_iris_scaled, method='ward')
    y_hierarchical = fcluster(Z_iris, t=3, criterion='maxclust')
    
    # Visualize with PCA
    pca_iris = PCA(n_components=2)
    X_iris_pca = pca_iris.fit_transform(X_iris_scaled)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # True labels
    axes[0].scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_iris, cmap='viridis', alpha=0.6, s=50)
    axes[0].set_title('True Species Labels')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    
    # K-Means
    axes[1].scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_kmeans, cmap='viridis', alpha=0.6, s=50)
    axes[1].set_title('K-Means Clustering')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    
    # Hierarchical
    axes[2].scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_hierarchical, cmap='viridis', alpha=0.6, s=50)
    axes[2].set_title('Hierarchical Clustering')
    axes[2].set_xlabel('PC1')
    axes[2].set_ylabel('PC2')
    
    # Feature importance (using cluster centroids)
    axes[3].barh(iris.feature_names, np.std(X_iris_scaled, axis=0))
    axes[3].set_title('Feature Standard Deviation')
    axes[3].set_xlabel('Std Dev (after scaling)')
    
    plt.tight_layout()
    plt.savefig('iris_clustering.png', dpi=150)
    plt.show()
    
    # Evaluate
    print("\nEvaluating clustering on Iris dataset:")
    labels_dict_iris = {
        'K-Means': y_kmeans,
        'Hierarchical': y_hierarchical
    }
    
    cluster_analysis(X_iris_scaled, labels_dict_iris)


# ============================================================
# Main Execution
# ============================================================

if __name__ == "__main__":
    print("Exercise 2.2: Clustering Analysis")
    print("=" * 60)
    
    # Part 1: K-Means from scratch
    X_scaled, y_true = kmeans_from_scratch_demo()
    
    # Part 2: Find optimal K
    inertias, silhouette_scores = find_optimal_k(X_scaled, max_k=8)
    
    # Part 3: Hierarchical clustering
    labels_hierarchical = hierarchical_clustering_demo(X_scaled)
    
    # Part 4: PCA visualization
    X_pca, pca = pca_visualization_demo(X_scaled, y_true)
    
    # Part 5: Compare clustering methods
    # Fit K-Means with sklearn for comparison
    from sklearn.cluster import KMeans
    kmeans_sklearn = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans_sklearn.fit(X_scaled)
    
    labels_dict = {
        'K-Means (Scratch)': KMeansScratch(n_clusters=4, random_state=42).fit(X_scaled).labels,
        'K-Means (sklearn)': kmeans_sklearn.labels_,
        'Hierarchical': labels_hierarchical
    }
    
    cluster_analysis(X_scaled, labels_dict)
    
    # Part 6: Real dataset application
    iris_clustering_application()
    
    print("\n" + "=" * 60)
    print("EXERCISE COMPLETE!")
    print("=" * 60)
    
    print("\nKey Learnings:")
    print("1. K-Means algorithm implementation from scratch")
    print("2. Hierarchical clustering and dendrogram interpretation")
    print("3. PCA for cluster visualization")
    print("4. Clustering evaluation metrics (Silhouette, DB, CH)")
    print("5. Finding optimal number of clusters")
    
    print("\nNext Steps:")
    print("- Explore DBSCAN for density-based clustering")
    print("- Try Gaussian Mixture Models (GMM)")
    print("- Experiment with different distance metrics")
    print("- Apply to real-world datasets")