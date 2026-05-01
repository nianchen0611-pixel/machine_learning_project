import numpy as np


# Not a standard keyword search, it's comparing the input embedding vector against all movies within the vector space.
def cosine_similarity(query_vector, movie_matrix):

    # length of query vector and movie vectors
    query_length = np.sqrt(np.sum(query_vector ** 2))
    movie_lengths = np.sqrt(np.sum(movie_matrix ** 2, axis=1))
    # dot product between the query vector and every movie vector
    dot_products = np.dot(movie_matrix, query_vector)

    # calculate the denominator of cosine similarity, avoid division by zero
    denominator = query_length * movie_lengths
    denominator = np.where(denominator == 0, 1e-10, denominator)

    # final calculation of cosine similarity
    similarities = dot_products / denominator

    return similarities





# Find the top k most similar movies.
def top_k_search(query_vector, movie_embeddings, k=10):

    # similarity score between the query and every movie
    scores = cosine_similarity(query_vector, movie_embeddings)

    # sort movie indices by similarity score from high to low
    top_indices = np.argsort(scores)[::-1][:k]
    # keep the only the first k scores
    top_scores = scores[top_indices]

    return top_indices, top_scores