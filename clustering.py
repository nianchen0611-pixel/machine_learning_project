import numpy as np


def initialize_centroids(X, k, random_state=42):
    rng = np.random.default_rng(random_state)
    indices = rng.choice(len(X), size=k, replace=False)
    return X[indices]


def assign_clusters(X, centroids):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)
    return labels


def update_centroids(X, labels, k, random_state=42):
    rng = np.random.default_rng(random_state)
    new_centroids = []

    for cluster_id in range(k):
        cluster_points = X[labels == cluster_id]

        if len(cluster_points) == 0:
            new_centroids.append(X[rng.integers(len(X))])
        else:
            new_centroids.append(cluster_points.mean(axis=0))

    return np.array(new_centroids)


def kmeans_from_scratch(X, k=5, max_iter=50, random_state=42):
    """
    K-Means implemented from scratch using NumPy.
    """
    centroids = initialize_centroids(X, k, random_state)

    for _ in range(max_iter):
        labels = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, labels, k, random_state)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids