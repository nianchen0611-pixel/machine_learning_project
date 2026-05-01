# Movie Detector: Find the Ideal Movie for You

## Project Overview

Movie Detector is a Streamlit web application that helps users find movies using natural language descriptions. Instead of searching only by exact movie titles or keywords, users can describe the type of movie they want to watch, such as "a science fiction movie about dreams" or "a romantic comedy about high school students." The app then returns movie recommendations based on semantic similarity.

This project uses several machine learning concepts from class, including text embeddings, cosine similarity, nearest-neighbor retrieval, K-Means clustering, and dimensionality reduction for visualization.

## Problem Statement

Movie platforms usually contain thousands of movies, and users may not always know the exact title they want to search for. Sometimes, users only know the mood, genre, or story idea they are interested in. Traditional keyword search may fail when the user's wording does not exactly match the movie description.

The main user-facing question is:
> What movie should I watch based on a natural language description?

## Dataset

This project uses the TMDB 5000 Movie Dataset from Kaggle. The dataset contains movie metadata such as:

- Movie title
- Overview
- Genres
- Keywords
- Release date
- Vote average
- Popularity

This project mainly uses the movie title, overview, genres, and keywords. These fields are combined into one searchable text field for each movie.

## Main Features

The web app includes the following features:

1. **Natural Language Movie Search**  
   Users can type a description of the movie they want to watch, and the app returns the most relevant movies.

2. **Similarity Score Ranking**  
   Each recommended movie is ranked using cosine similarity between the user query embedding and movie embeddings.

3. **Find Similar Movies by Title**  
   Users can enter a movie title, and the app finds movies that are semantically similar to that movie.

4. **Flexible Title Matching**  
   The app normalizes movie titles, so inputs like "spider man" can still match "Spider-Man."

5. **K-Means Clustering from Scratch**  
   K-Means clustering is implemented manually using NumPy. The app uses this clustering result in the movie embedding visualization.

6. **2D Movie Embedding Visualization**  
   The app uses PCA to reduce high-dimensional movie embeddings into two dimensions and visualizes movie clusters.

## Machine Learning Methods

### Text Embeddings

Each movie is represented by a text field that combines its title, overview, genres, and keywords. A sentence-transformer model converts this text into a dense vector embedding. The user's search query is also converted into the same embedding space.

### Cosine Similarity Retrieval

After converting the user query into an embedding, the app computes cosine similarity between the query vector and all movie vectors. Movies with higher similarity scores are considered more relevant to the user's input.

The cosine similarity and top-k retrieval logic are implemented in `retrieval.py`.

### K-Means Clustering from Scratch

To satisfy the course requirement that at least one course algorithm must be implemented from scratch, I implemented K-Means clustering manually in `clustering.py`.

Our K-Means implementation includes:

- Random centroid initialization
- Assigning each movie embedding to the nearest centroid
- Updating centroids by averaging assigned points
- Repeating the process until convergence or a maximum number of iterations

I did not use `sklearn.cluster.KMeans` for this part. The clustering result is used in the 2D visualization of movie embeddings.

### PCA Visualization

The original movie embeddings are high-dimensional, so PCA is used to reduce them to two dimensions for visualization. This helps users see how movies are grouped in the embedding space.

## Project Structure

```text
movie_finder/
│
├── app.py
├── data.py
├── embed.py
├── retrieval.py
├── clustering.py
├── utils.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── data/
    └── movies.csv