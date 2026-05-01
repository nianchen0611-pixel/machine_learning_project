from sentence_transformers import SentenceTransformer
import numpy as np
import streamlit as st


# Streamlit 的缓存功能，不用每次都重复计算 embeddings
@st.cache_resource
# pretrained sentence-transformer model，专门用来把文本转换成embedding vector的模型
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# 直接使用缓存里的模型 cached model，速度快很多
@st.cache_data
def compute_movie_embeddings(texts):
    model = load_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=False)
    return np.array(embeddings)


def compute_query_embedding(query):
    model = load_embedding_model()
    embedding = model.encode([query], show_progress_bar=False)
    return np.array(embedding[0])