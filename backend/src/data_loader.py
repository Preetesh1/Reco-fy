# backend/src/data_loader.py

import os
import requests
import zipfile
import pandas as pd
import numpy as np
from io import BytesIO

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
RAW_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


# ─────────────────────────────────────────────
# STEP 1: DOWNLOAD & EXTRACT
# ─────────────────────────────────────────────
def download_movielens(force=False):
    """
    Downloads and extracts the MovieLens 100K dataset.
    
    Why ml-100k?
    - 100,000 ratings from 943 users on 1,682 movies
    - Clean, well-structured, industry-standard benchmark
    - Small enough to train fast, large enough to be meaningful
    
    Args:
        force (bool): Re-download even if data already exists
    """
    target_dir = os.path.join(RAW_DATA_DIR, "ml-100k")

    if os.path.exists(target_dir) and not force:
        print("✅ Dataset already exists. Skipping download.")
        return target_dir

    print("⬇️  Downloading MovieLens 100K...")
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    response = requests.get(MOVIELENS_URL, stream=True)
    response.raise_for_status()

    with zipfile.ZipFile(BytesIO(response.content)) as z:
        z.extractall(RAW_DATA_DIR)

    print(f"✅ Dataset extracted to: {target_dir}")
    return target_dir


# ─────────────────────────────────────────────
# STEP 2: LOAD RAW RATINGS
# ─────────────────────────────────────────────
def load_ratings(data_dir=None):
    """
    Loads the u.data file — the core ratings file.

    File format (tab-separated, no header):
        user_id | item_id | rating | timestamp

    Rating scale: 1–5 stars
    Total rows: 100,000

    Returns:
        pd.DataFrame with columns: user_id, movie_id, rating, timestamp
    """
    if data_dir is None:
        data_dir = os.path.join(RAW_DATA_DIR, "ml-100k")

    filepath = os.path.join(data_dir, "u.data")

    df = pd.read_csv(
        filepath,
        sep="\t",
        names=["user_id", "movie_id", "rating", "timestamp"],
        encoding="latin-1"
    )

    # Convert timestamp (Unix epoch) to readable datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

    print(f"✅ Ratings loaded: {df.shape[0]:,} rows, {df['user_id'].nunique()} users, "
          f"{df['movie_id'].nunique()} movies")

    return df


# ─────────────────────────────────────────────
# STEP 3: LOAD MOVIE METADATA
# ─────────────────────────────────────────────
def load_movies(data_dir=None):
    """
    Loads u.item — movie titles and genre flags.

    The genre columns are binary (0/1) one-hot encoded flags.
    There are 19 genres total in this dataset.

    Returns:
        pd.DataFrame with movie_id, title, release_date, and 19 genre columns
    """
    if data_dir is None:
        data_dir = os.path.join(RAW_DATA_DIR, "ml-100k")

    filepath = os.path.join(data_dir, "u.item")

    genre_columns = [
        "unknown", "Action", "Adventure", "Animation", "Children",
        "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
        "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
        "Sci-Fi", "Thriller", "War", "Western"
    ]

    columns = ["movie_id", "title", "release_date", "video_release_date",
               "imdb_url"] + genre_columns

    df = pd.read_csv(
        filepath,
        sep="|",
        names=columns,
        encoding="latin-1",
        usecols=["movie_id", "title", "release_date"] + genre_columns
    )

    # Parse release year from title string e.g. "Toy Story (1995)"
    df["year"] = df["title"].str.extract(r"\((\d{4})\)").astype("Int64")

    print(f"✅ Movies loaded: {df.shape[0]:,} movies")
    return df


# ─────────────────────────────────────────────
# STEP 4: LOAD USER DEMOGRAPHICS
# ─────────────────────────────────────────────
def load_users(data_dir=None):
    """
    Loads u.user — user demographic info.

    Columns: user_id, age, gender, occupation, zip_code
    Useful later for: demographic-based cold start fallback (Day 4)

    Returns:
        pd.DataFrame with user demographic data
    """
    if data_dir is None:
        data_dir = os.path.join(RAW_DATA_DIR, "ml-100k")

    filepath = os.path.join(data_dir, "u.user")

    df = pd.read_csv(
        filepath,
        sep="|",
        names=["user_id", "age", "gender", "occupation", "zip_code"],
        encoding="latin-1"
    )

    print(f"✅ Users loaded: {df.shape[0]:,} users")
    return df


# ─────────────────────────────────────────────
# STEP 5: BUILD USER-ITEM MATRIX
# ─────────────────────────────────────────────
def build_user_item_matrix(ratings_df):
    """
    Pivots the ratings DataFrame into a User-Item matrix.

    Shape: (943 users) × (1,682 movies)
    
    WHY THIS MATTERS:
    This matrix is the core data structure for collaborative filtering.
    
    - Rows = users
    - Columns = movies  
    - Cell value = rating (1–5), or 0 if not rated
    
    Sparsity: ~93.7% of cells are 0 (unrated).
    This sparsity is exactly what the ML models learn to fill in.

    Returns:
        pd.DataFrame (user-item matrix, NaN filled with 0)
    """
    matrix = ratings_df.pivot_table(
        index="user_id",
        columns="movie_id",
        values="rating"
    ).fillna(0)

    sparsity = 1 - (ratings_df.shape[0] / (matrix.shape[0] * matrix.shape[1]))
    print(f"✅ User-Item Matrix built: {matrix.shape}")
    print(f"   Sparsity: {sparsity:.1%} of cells are unrated")

    return matrix


# ─────────────────────────────────────────────
# STEP 6: TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
def split_data(ratings_df, test_size=0.2, random_state=42):
    """
    Splits ratings into train and test sets.

    WHY NOT sklearn's train_test_split directly on the matrix?
    Because we split at the RATING level, not the user level.
    This simulates: "hold out 20% of all ratings, train on the rest."

    Args:
        ratings_df: Full ratings DataFrame
        test_size: Fraction to hold out for testing
        random_state: For reproducibility

    Returns:
        train_df, test_df
    """
    from sklearn.model_selection import train_test_split

    train_df, test_df = train_test_split(
        ratings_df,
        test_size=test_size,
        random_state=random_state,
        stratify=ratings_df["rating"]  # preserve rating distribution
    )

    print(f"✅ Split complete → Train: {len(train_df):,} | Test: {len(test_df):,}")
    return train_df, test_df


# ─────────────────────────────────────────────
# STEP 7: DATA QUALITY CHECKS
# ─────────────────────────────────────────────
def validate_data(ratings_df, movies_df, users_df):
    """
    Runs basic sanity checks on all three DataFrames.
    Catches issues early before they silently corrupt model training.
    """
    issues = []

    # Check rating range
    invalid_ratings = ratings_df[~ratings_df["rating"].between(1, 5)]
    if not invalid_ratings.empty:
        issues.append(f"⚠️  {len(invalid_ratings)} ratings outside 1–5 range")

    # Check for duplicate ratings (same user, same movie)
    dupes = ratings_df.duplicated(subset=["user_id", "movie_id"])
    if dupes.any():
        issues.append(f"⚠️  {dupes.sum()} duplicate user-movie rating pairs")

    # Check all rated movie_ids exist in movies_df
    rated_movies = set(ratings_df["movie_id"].unique())
    known_movies = set(movies_df["movie_id"].unique())
    missing = rated_movies - known_movies
    if missing:
        issues.append(f"⚠️  {len(missing)} rated movies missing from movies metadata")

    if issues:
        for issue in issues:
            print(issue)
    else:
        print("✅ All data validation checks passed")

    return len(issues) == 0


# ─────────────────────────────────────────────
# CONVENIENCE: load everything at once
# ─────────────────────────────────────────────
def load_all():
    """
    Single call to get all processed data.
    This is what Day 2+ modules will import.

    Returns:
        ratings_df, movies_df, users_df, user_item_matrix, train_df, test_df
    """
    download_movielens()
    ratings_df = load_ratings()
    movies_df = load_movies()
    users_df = load_users()
    validate_data(ratings_df, movies_df, users_df)
    user_item_matrix = build_user_item_matrix(ratings_df)
    train_df, test_df = split_data(ratings_df)

    return ratings_df, movies_df, users_df, user_item_matrix, train_df, test_df


if __name__ == "__main__":
    ratings, movies, users, matrix, train, test = load_all()

    print("\n── Sample Ratings ──")
    print(ratings.head())

    print("\n── Sample Movies ──")
    print(movies[["movie_id", "title", "year"]].head())

    print("\n── Matrix Shape ──")
    print(f"Users: {matrix.shape[0]}, Movies: {matrix.shape[1]}")