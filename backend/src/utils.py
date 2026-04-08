# backend/src/utils.py

import pandas as pd
import numpy as np


def get_movies_rated_by_user(user_id, ratings_df):
    """
    Returns all movies a user has rated, sorted by rating desc.
    Used for: display, cold-start checks, evaluation.
    """
    user_ratings = ratings_df[ratings_df["user_id"] == user_id]
    return user_ratings.sort_values("rating", ascending=False)


def get_unrated_movies(user_id, ratings_df, movies_df):
    """
    Returns movies the user has NOT yet rated.
    These are the candidates for recommendation.

    WHY: We never recommend something a user already rated.
    That would be pointless and hurt perceived quality.
    """
    rated_ids = set(ratings_df[ratings_df["user_id"] == user_id]["movie_id"])
    all_ids = set(movies_df["movie_id"])
    unrated_ids = all_ids - rated_ids
    return movies_df[movies_df["movie_id"].isin(unrated_ids)]


def enrich_recommendations(rec_df, movies_df):
    """
    Joins recommendation scores with movie metadata (title, genres, year).
    Every recommender returns (movie_id, score) — this adds the human-readable info.

    Args:
        rec_df: DataFrame with columns [movie_id, predicted_rating]
        movies_df: Full movies metadata DataFrame

    Returns:
        Enriched DataFrame with title, year, genres, score
    """
    genre_cols = [
        "Action", "Adventure", "Animation", "Children", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western"
    ]

    # Build genre string per movie e.g. "Action | Thriller"
    movies_copy = movies_df.copy()
    movies_copy["genres"] = movies_copy[genre_cols].apply(
        lambda row: " | ".join([g for g in genre_cols if row[g] == 1]),
        axis=1
    )

    enriched = rec_df.merge(
        movies_copy[["movie_id", "title", "year", "genres"]],
        on="movie_id",
        how="left"
    ).sort_values("predicted_rating", ascending=False)

    return enriched


def print_recommendations(enriched_df, user_id, method="k-NN"):
    """Pretty-prints recommendations to terminal."""
    print(f"\n{'═'*60}")
    print(f"  🎬 Top Recommendations for User {user_id}  [{method}]")
    print(f"{'═'*60}")
    for i, row in enriched_df.iterrows():
        print(f"  {row.name+1 if hasattr(row,'name') else '·'}. {row['title']} ({row['year']})")
        print(f"     Predicted Rating: {'⭐' * round(row['predicted_rating'])} "
              f"({row['predicted_rating']:.2f})")
        print(f"     Genres: {row['genres']}")
        print()