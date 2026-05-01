import pandas as pd
import ast


def clean_json_list(value):
    """
    Convert TMDB JSON-like list strings into readable names.
    Example: [{"id": 28, "name": "Action"}] -> Action
    """
    if pd.isna(value):
        return ""

    try:
        items = ast.literal_eval(value)
        if isinstance(items, list):
            return ", ".join([item.get("name", "") for item in items])
    except Exception:
        return str(value)

    return str(value)


def load_movies(file_path="data/movies.csv", max_rows=1000):
    """
    Load and clean the TMDB movie dataset.
    """
    df = pd.read_csv(file_path)

    required_cols = [
        "title",
        "overview",
        "genres",
        "keywords",
        "release_date",
        "vote_average",
        "popularity"
    ]

    available_cols = [col for col in required_cols if col in df.columns]
    df = df[available_cols].copy()

    df = df.dropna(subset=["title", "overview"])

    if "genres" in df.columns:
        df["genres_clean"] = df["genres"].apply(clean_json_list)
    else:
        df["genres_clean"] = ""

    if "keywords" in df.columns:
        df["keywords_clean"] = df["keywords"].apply(clean_json_list)
    else:
        df["keywords_clean"] = ""

    if "release_date" in df.columns:
        df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    else:
        df["year"] = ""

    df["search_text"] = (
        df["title"].fillna("") + ". " +
        df["overview"].fillna("") + ". " +
        df["genres_clean"].fillna("") + ". " +
        df["keywords_clean"].fillna("")
    )

    df = df.head(max_rows).reset_index(drop=True)

    return df