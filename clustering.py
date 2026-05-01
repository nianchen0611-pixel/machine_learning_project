import numpy as np



def initialize_centroids(X, k, random_state=42):

    # Create a random number generator with a fixed seed, reproducible.
    rng = np.random.default_rng(random_state)
    # Choose k different movie embeddings as the initial centroids.
    indices = rng.choice(len(X), size=k, replace=False)

    return X[indices]


def assign_clusters(X, centroids):

    # Compute the distance between every movie embedding and every centroid.
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    # Assign each movie to the nearest centroid, the returned label is the cluster number for each movie.
    labels = np.argmin(distances, axis=1)

    return labels


def update_centroids(X, labels, k, random_state=42):

    # Create a random number generator for handling empty clusters.
    rng = np.random.default_rng(random_state)
    # Store the updated centroid for each cluster.
    new_centroids = []

    for cluster_id in range(k):

        # Select all movie embeddings assigned to the current cluster.
        cluster_points = X[labels == cluster_id]
        # Prevents the algorithm from losing a cluster.
        if len(cluster_points) == 0:
            # If a cluster has no movies, randomly choose one movie embedding as its new centroid.
            new_centroids.append(X[rng.integers(len(X))])

        else:
            # If the cluster has assigned, update centroid.
            new_centroids.append(cluster_points.mean(axis=0))


    return np.array(new_centroids)


# Each movie became a vector, movie_embeddings as X, 
def kmeans_from_scratch(X, k=5, max_iter=50, random_state=42):
    """
    K-Means implemented from scratch using NumPy.
    """

    # Randomly select 5 movie embeddings as initial centroids.
    centroids = initialize_centroids(X, k, random_state)
    for _ in range(max_iter):

        # Calculate its distance to each centroid, then assign to the nearest centroid.
        labels = assign_clusters(X, centroids)

        # Update centroids based on the current cluster assignments.
        new_centroids = update_centroids(X, labels, k, random_state)

        if np.allclose(centroids, new_centroids):
            break

        centroids = new_centroids

    return labels, centroids