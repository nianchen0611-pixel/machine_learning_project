from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st


# Caching feature, avoid recomputing embeddings every time.
@st.cache_resource
def load_embedding_model():
    # Pretrained sentence-transformer model
    return SentenceTransformer("all-MiniLM-L6-v2")


# Directly use cached model, much faster
@st.cache_data
def compute_movie_embeddings(texts):
    # Convert all movie search texts into embedding vectors.
    model = load_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings)

# Convert the user's search query into one embedding vector.
def compute_query_embedding(query):
    model = load_embedding_model()
    embedding = model.encode([query], show_progress_bar=False)
    return np.array(embedding[0])