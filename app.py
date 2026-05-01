import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

from data import load_movies
from embed import compute_movie_embeddings, compute_query_embedding
from retrieval import top_k_search, cosine_similarity
from clustering import kmeans_from_scratch
from utils import shorten_text, normalize_title



st.set_page_config(
    page_title="Movie Finder",
    page_icon="🎬",
    layout="wide"
)


st.title("🎬 Movie Detector: Find the ideal movie for you")
st.write(
    """
    This web app helps users find movies by natural language descriptions.
    It uses text embeddings, cosine similarity retrieval, and a K-Means algorithm
    implemented from scratch.
    """
)


# Initialize session states for search sections
if "movie_search_active" not in st.session_state:
    st.session_state.movie_search_active = False

if "similar_search_active" not in st.session_state:
    st.session_state.similar_search_active = False

if "movie_query_input" not in st.session_state:
    st.session_state.movie_query_input = ""

if "similar_title_input" not in st.session_state:
    st.session_state.similar_title_input = ""

if "movie_top_indices" not in st.session_state:
    st.session_state.movie_top_indices = None

if "movie_top_scores" not in st.session_state:
    st.session_state.movie_top_scores = None

if "similar_selected_idx" not in st.session_state:
    st.session_state.similar_selected_idx = None

if "similar_indices" not in st.session_state:
    st.session_state.similar_indices = None

if "similar_scores" not in st.session_state:
    st.session_state.similar_scores = None

# Makes the input box and recommendation results return to an empty state.
def clear_movie_search():
    
    # Clear all stored results from the main natural language search section.
    st.session_state.movie_search_active = False
    st.session_state.movie_query_input = ""
    st.session_state.movie_top_indices = None
    st.session_state.movie_top_scores = None


# Removes the matched movie, similar movie results, and the input title.
def clear_similar_search():
    
    # Clear all stored results from the similar movie search section.
    st.session_state.similar_search_active = False
    st.session_state.similar_title_input = ""
    st.session_state.similar_selected_idx = None
    st.session_state.similar_indices = None
    st.session_state.similar_scores = None


st.sidebar.header("Settings")
max_rows = st.sidebar.slider("Number of movies to load", 200, 2000, 1000, step=100)
top_k = st.sidebar.slider("Number of recommendations", 5, 20, 10)
k_clusters = 5


with st.spinner("Loading movie data..."):
    movies = load_movies("data/movies.csv", max_rows=max_rows)

# Use SentenceTransformer model to convert each movie's text into an embedding vector.
with st.spinner("Computing movie embeddings..."):
    # total number of movie vectors is saved 
    movie_embeddings = compute_movie_embeddings(movies["search_text"].tolist())

# Pass all movie_embeddings, info of all movies in vector, into k-means, and then group them with similar semantics into the same group.
with st.spinner("Running K-Means clustering from scratch..."):
    cluster_labels, centroids = kmeans_from_scratch(
        movie_embeddings,
        k=k_clusters,
        max_iter=50,
        random_state=42
    )

movies["cluster"] = cluster_labels



# Main natural language search
# =========================

st.subheader("🔎 Search for Movies: ")


# User text-input box
query = st.text_input(
    "Describe the movie you want to search:",
    placeholder="Example: a science fiction movie about dreams",
    key="movie_query_input"
)

col_search, col_space, col_clear_search = st.columns([1.2, 7.2, 0.9])


# Start button
with col_search:
    search_clicked = st.button("Search Movies")

# Clear button
with col_clear_search:
    st.button(
        "✕ Clear Results",
        disabled=not st.session_state.movie_search_active,
        on_click=clear_movie_search,
        key="clear_movie_search_button",
        use_container_width=True
    )


if search_clicked:
    # Clear previous movie search results first
    st.session_state.movie_search_active = False
    st.session_state.movie_top_indices = None
    st.session_state.movie_top_scores = None

    if query.strip() == "":
        st.warning("Please enter a movie description first.")
    else:
        query_embedding = compute_query_embedding(query)

        top_indices, top_scores = top_k_search(
            query_embedding,
            movie_embeddings,
            k=top_k
        )

        # Minimum similarity score threshold
        # If the best result is below this value, we treat it as a weak match.
        min_similarity_score = 0.35

        if len(top_scores) == 0 or top_scores[0] < min_similarity_score:
            st.warning(
                "No strong movie match found. Please try a more meaningful movie description."
            )
        else:
            st.session_state.movie_search_active = True
            st.session_state.movie_top_indices = top_indices
            st.session_state.movie_top_scores = top_scores

            # Rerun so the Clear button immediately becomes clickable
            st.rerun()

# Movie search result display
if st.session_state.movie_search_active:
    st.subheader("Top Movie Recommendations:")

    for rank, (idx, score) in enumerate(
        zip(st.session_state.movie_top_indices, st.session_state.movie_top_scores),
        start=1
    ):
        movie = movies.iloc[idx]

        with st.container(border=True):
            st.markdown(f"### {rank}. {movie['title']}")

            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Similarity Score:** {score:.3f}")

            with col2:
                if "year" in movie and pd.notna(movie["year"]):
                    st.write(f"**Year:** {int(movie['year'])}")
                else:
                    st.write("**Year:** Unknown")

            st.write(f"**Genres:** {movie['genres_clean']}")
            st.write(f"**Keywords:** {movie['keywords_clean']}")
            st.write(f"**Overview:** {shorten_text(movie['overview'])}")



# Similar movie search by title
# =========================

st.subheader("🎞️ Find Similar Movies by Title: ")

movie_title_input = st.text_input(
    "Type a movie title:",
    placeholder="Example: Spider man, Spider-Man, The Dark Knight",
    key="similar_title_input"
)

col_find, col_space, col_clear_similar = st.columns([1.2, 7.2, 0.9])

# Start button
with col_find:
    find_similar_clicked = st.button("Find Similar Movies")

# Clear button
with col_clear_similar:
    st.button(
        "✕ Clear Results",
        disabled=not st.session_state.similar_search_active,
        on_click=clear_similar_search,
        key="clear_similar_search_button",
        use_container_width=True
    )


if find_similar_clicked:

    # Clear previous similar movie results first
    st.session_state.similar_search_active = False
    st.session_state.similar_selected_idx = None
    st.session_state.similar_indices = None
    st.session_state.similar_scores = None

    if movie_title_input.strip() == "":
        st.warning("Please enter a movie title first.")

    else:
        # Normalize user input
        normalized_input = normalize_title(movie_title_input)

        # If the input only contains symbols or punctuation, normalization becomes empty.
        # Example: "!!!!!!" -> ""
        if normalized_input == "":
            st.warning("Please enter a valid movie title.")

        else:
            # Create a normalized title column for flexible matching
            movies["normalized_title"] = movies["title"].apply(normalize_title)

            # First try exact matching after normalization
            exact_matches = movies[movies["normalized_title"] == normalized_input]

            # If exact match exists, use it first
            if not exact_matches.empty:
                matched_movies = exact_matches

            # If no exact match, then try partial matching
            else:
                matched_movies = movies[
                    movies["normalized_title"].str.contains(normalized_input, na=False)
                ]

            if matched_movies.empty:
                st.warning("No matching movie title found. Try another title.")

            else:
                # Use the best matched movie
                selected_idx = matched_movies.index[0]
                selected_vector = movie_embeddings[selected_idx]

                # Compare the selected movie with all other movies
                scores = cosine_similarity(selected_vector, movie_embeddings)

                # Skip the first result because it is the selected movie itself
                similar_indices = scores.argsort()[::-1][1:top_k + 1]

                st.session_state.similar_search_active = True
                st.session_state.similar_selected_idx = selected_idx
                st.session_state.similar_indices = similar_indices
                st.session_state.similar_scores = scores

                # Rerun so the Clear button immediately becomes clickable
                st.rerun()


if st.session_state.similar_search_active:
    selected_idx = st.session_state.similar_selected_idx
    selected_movie = movies.loc[selected_idx]
    selected_movie_title = movies.loc[selected_idx, "title"]
    scores = st.session_state.similar_scores
    similar_indices = st.session_state.similar_indices

    # First show the matched movie itself
    st.success(f"Matched Movie Found: {selected_movie_title}")

    with st.container(border=True):
        st.markdown(f"### {selected_movie['title']}")
        st.write("**Similarity Score:** 1.000")
        st.write(f"**Genres:** {selected_movie['genres_clean']}")
        st.write(f"**Overview:** {shorten_text(selected_movie['overview'], 250)}")

    # Then show similar movies
    st.warning(f"Similar Movies Found for: {selected_movie_title}")

    for idx in similar_indices:
        movie = movies.iloc[idx]

        with st.container(border=True):
            st.markdown(f"### {movie['title']}")
            st.write(f"**Similarity Score:** {scores[idx]:.3f}")
            st.write(f"**Genres:** {movie['genres_clean']}")
            st.write(f"**Overview:** {shorten_text(movie['overview'], 250)}")



# Visualization
# =========================

st.subheader("📊 Movie Embedding Visualization")
st.caption("K-Means clustering is implemented from scratch and used to color the movie embedding visualization.")

if st.checkbox("Show 2D cluster visualization"):
    with st.spinner("Reducing embeddings to 2D using PCA..."):
        pca = PCA(n_components=2)
        coords = pca.fit_transform(movie_embeddings)

    plot_df = pd.DataFrame({
        "x": coords[:, 0],
        "y": coords[:, 1],
        "title": movies["title"],
        "cluster": movies["cluster"].astype(str),
        "genres": movies["genres_clean"]
    })

    fig = px.scatter(
        plot_df,
        x="x",
        y="y",
        color="cluster",
        hover_data=["title", "genres"],
        title="2D Visualization of Movie Embeddings by K-Means Cluster"
    )

    st.plotly_chart(fig, use_container_width=True)



# Explanation
# =========================

st.subheader("🧠 How It Works")

st.write(
    """
    The app turns movie descriptions and user inputs into text embeddings.  
    It then uses cosine similarity to find the most relevant movies.  
    K-Means clustering is implemented from scratch to group movies by embedding similarity.
    """
)