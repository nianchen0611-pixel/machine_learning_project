import numpy as np


# 不是普通关键词keyword搜索，是对比输入embedding vector和所有电影在向量空间vector space的对比度 
# 给所有电影打分，找到最接近的电影
def cosine_similarity(query_vector, movie_matrix):

    '''
    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)

    denominator = query_norm * matrix_norms
    denominator = np.where(denominator == 0, 1e-10, denominator)

    similarities = np.dot(matrix, query_vector) / denominator
    return similarities


    
    '''

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





# 找出最相似的前 k 部电影
def top_k_search(query_vector, movie_embeddings, k=10):

    # similarity score between the query and every movie
    scores = cosine_similarity(query_vector, movie_embeddings)

    # sort movie indices by similarity score from high to low
    top_indices = np.argsort(scores)[::-1][:k]

    # keep the only the first k scores 找到最接近的k部电影
    top_scores = scores[top_indices]


    return top_indices, top_scores