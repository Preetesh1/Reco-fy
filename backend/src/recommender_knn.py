# backend/src/recommender_knn.py

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CLASS: UserBasedKNN
# ─────────────────────────────────────────────

class UserBasedKNN:
    """
    User-Based Collaborative Filtering using cosine similarity.

    ALGORITHM:
    1. Build user-item matrix (ratings or 0)
    2. Compute cosine similarity between ALL pairs of users
    3. For a target user:
       a. Find K most similar users who have rated the target movie
       b. Predict rating = weighted average of their ratings
          (weight = similarity score)
    4. Rank all unrated movies by predicted rating → recommendations

    COMPLEXITY:
    - Training: O(U²) where U = number of users (943² ≈ 889K ops — fast)
    - Prediction: O(K × M) where M = movies
    """

    def __init__(self, k=20, min_common_ratings=3):
        """
        Args:
            k (int): Number of nearest neighbors to consider.
                     Too low → noisy, too high → dilutes similarity signal.
                     20 is a well-tested default for ml-100k.

            min_common_ratings (int): Minimum movies two users must have
                     both rated to be considered neighbors.
                     Prevents spurious similarity from 1 shared rating.
        """
        self.k = k
        self.min_common_ratings = min_common_ratings
        self.user_item_matrix = None
        self.similarity_matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.is_fitted = False

    def fit(self, ratings_df):
        """
        Builds the user-item matrix and computes user similarity.

        WHY normalize before cosine similarity?
        cosine_similarity from sklearn already normalizes internally,
        but we store the normalized matrix for fast dot-product
        similarity lookups in predict().

        Args:
            ratings_df: DataFrame with [user_id, movie_id, rating]
        """
        print("🔧 Building User-Item matrix...")

        # Pivot to matrix: rows=users, cols=movies, values=ratings
        matrix = ratings_df.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating"
        ).fillna(0)

        self.user_ids = matrix.index.tolist()
        self.movie_ids = matrix.columns.tolist()
        self.user_item_matrix = matrix

        print(f"   Matrix shape: {matrix.shape}")
        print("🔧 Computing user-user cosine similarity...")

        # Convert to numpy for sklearn
        matrix_values = matrix.values

        # cosine_similarity returns an (N_users × N_users) matrix
        # similarity_matrix[i][j] = cosine similarity between user i and user j
        # Diagonal = 1.0 (every user is identical to themselves)
        self.similarity_matrix = cosine_similarity(matrix_values)

        # Zero out self-similarity (diagonal) so a user isn't
        # their own nearest neighbor
        np.fill_diagonal(self.similarity_matrix, 0)

        self.is_fitted = True
        print(f"✅ UserBasedKNN fitted. Similarity matrix: "
              f"{self.similarity_matrix.shape}")
        return self

    def _get_similar_users(self, user_id):
        """
        Returns the K most similar users to the given user_id,
        sorted by similarity descending.

        Returns:
            List of (user_id, similarity_score) tuples
        """
        if user_id not in self.user_ids:
            raise ValueError(f"User {user_id} not in training data.")

        user_idx = self.user_ids.index(user_id)
        similarities = self.similarity_matrix[user_idx]

        # Get indices of top-K most similar users
        top_k_indices = np.argsort(similarities)[::-1][:self.k]

        return [
            (self.user_ids[idx], similarities[idx])
            for idx in top_k_indices
            if similarities[idx] > 0
        ]

    def predict_rating(self, user_id, movie_id):
        """
        Predicts the rating user_id would give movie_id.

        FORMULA (weighted average):

                 Σ sim(u, v) × rating(v, movie)
        pred =  ────────────────────────────────
                       Σ |sim(u, v)|

        where v = neighbor who has rated movie_id

        WHY weighted average?
        A neighbor with similarity 0.9 should influence the
        prediction more than one with similarity 0.1.

        Returns:
            float: predicted rating (1.0–5.0), or None if unpredictable
        """
        similar_users = self._get_similar_users(user_id)

        numerator = 0
        denominator = 0
        neighbors_used = 0

        for neighbor_id, sim_score in similar_users:
            neighbor_rating = self.user_item_matrix.loc[neighbor_id, movie_id] \
                if movie_id in self.movie_ids else 0

            if neighbor_rating > 0:  # neighbor actually rated this movie
                numerator += sim_score * neighbor_rating
                denominator += abs(sim_score)
                neighbors_used += 1

        if denominator == 0 or neighbors_used < self.min_common_ratings:
            return None  # Not enough signal → cold start case

        predicted = numerator / denominator
        # Clip to valid rating range
        return float(np.clip(predicted, 1.0, 5.0))

    def recommend(self, user_id, n=10, ratings_df=None, movies_df=None):
        """
        Generates top-N movie recommendations for a user.

        Process:
        1. Get all movies the user hasn't rated
        2. Predict rating for each
        3. Sort by predicted rating descending
        4. Return top N

        Args:
            user_id (int): Target user
            n (int): Number of recommendations
            ratings_df: Needed to find already-rated movies
            movies_df: Needed to filter valid movie IDs

        Returns:
            pd.DataFrame with [movie_id, predicted_rating]
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        # Movies this user already rated — exclude from recommendations
        if ratings_df is not None:
            rated_ids = set(
                ratings_df[ratings_df["user_id"] == user_id]["movie_id"]
            )
        else:
            rated_ids = set(
                self.user_item_matrix.columns[
                    self.user_item_matrix.loc[user_id] > 0
                ]
            )

        print(f"🔍 Generating recommendations for User {user_id}...")
        print(f"   Already rated: {len(rated_ids)} movies")

        predictions = []
        candidate_movies = [m for m in self.movie_ids if m not in rated_ids]

        for movie_id in candidate_movies:
            pred = self.predict_rating(user_id, movie_id)
            if pred is not None:
                predictions.append({
                    "movie_id": movie_id,
                    "predicted_rating": round(pred, 3)
                })

        if not predictions:
            print("⚠️  No predictions generated — user may have too few ratings")
            return pd.DataFrame(columns=["movie_id", "predicted_rating"])

        result_df = (
            pd.DataFrame(predictions)
            .sort_values("predicted_rating", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )

        print(f"✅ Generated {len(result_df)} recommendations")
        return result_df


# ─────────────────────────────────────────────
# CLASS: ItemBasedKNN
# ─────────────────────────────────────────────

class ItemBasedKNN:
    """
    Item-Based Collaborative Filtering using cosine similarity.

    ALGORITHM:
    1. Build item-user matrix (transpose of user-item)
    2. Compute cosine similarity between ALL pairs of items
    3. For a target user + candidate movie:
       a. Find K movies most similar to candidate that user HAS rated
       b. Predict rating = weighted average of user's ratings
          on those similar movies
    4. Rank all candidates → recommendations

    WHY item-based is often better:
    - Item similarity is more stable over time than user similarity
    - Users' tastes drift; a movie's "neighborhood" doesn't change
    - Scales better for large user bases (precompute item similarity once)

    Amazon's original recommendation engine was item-based CF.
    """

    def __init__(self, k=20):
        self.k = k
        self.item_similarity_matrix = None
        self.user_item_matrix = None
        self.user_ids = None
        self.movie_ids = None
        self.movie_id_to_idx = {}
        self.is_fitted = False

    def fit(self, ratings_df):
        """
        Builds item-item similarity matrix.
        Transpose of user-item → item-user → compute cosine similarity.
        """
        print("🔧 Building Item-User matrix...")

        matrix = ratings_df.pivot_table(
            index="user_id",
            columns="movie_id",
            values="rating"
        ).fillna(0)

        self.user_ids = matrix.index.tolist()
        self.movie_ids = matrix.columns.tolist()
        self.movie_id_to_idx = {mid: idx for idx, mid in enumerate(self.movie_ids)}
        self.user_item_matrix = matrix

        print(f"   Matrix shape: {matrix.shape}")
        print("🔧 Computing item-item cosine similarity...")

        # Transpose: now rows=movies, cols=users
        # Each movie is a vector of ratings it received from all users
        item_matrix = matrix.values.T

        self.item_similarity_matrix = cosine_similarity(item_matrix)
        np.fill_diagonal(self.item_similarity_matrix, 0)

        self.is_fitted = True
        print(f"✅ ItemBasedKNN fitted. Item similarity matrix: "
              f"{self.item_similarity_matrix.shape}")
        return self

    def predict_rating(self, user_id, movie_id):
        """
        Predicts the rating user_id would give movie_id.

        LOGIC:
        - Find K items most similar to movie_id
        - Filter to items user_id HAS already rated
        - Weighted average of those ratings

        This answers: "Given what this user liked/disliked,
        how much would they like this new movie?"
        """
        if movie_id not in self.movie_id_to_idx:
            return None

        movie_idx = self.movie_id_to_idx[movie_id]
        similarities = self.item_similarity_matrix[movie_idx]

        # Get top-K similar items
        top_k_indices = np.argsort(similarities)[::-1][:self.k]

        # Get what this user rated among similar items
        if user_id not in self.user_ids:
            return None

        user_ratings = self.user_item_matrix.loc[user_id]

        numerator = 0
        denominator = 0

        for idx in top_k_indices:
            similar_movie_id = self.movie_ids[idx]
            sim_score = similarities[idx]
            user_rating_for_similar = user_ratings.get(similar_movie_id, 0)

            if user_rating_for_similar > 0:
                numerator += sim_score * user_rating_for_similar
                denominator += abs(sim_score)

        if denominator == 0:
            return None

        return float(np.clip(numerator / denominator, 1.0, 5.0))

    def recommend(self, user_id, n=10, ratings_df=None):
        """
        Same interface as UserBasedKNN.recommend() — intentional.
        Day 5 Flask API will call either recommender transparently.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before recommend()")

        if ratings_df is not None:
            rated_ids = set(
                ratings_df[ratings_df["user_id"] == user_id]["movie_id"]
            )
        else:
            rated_ids = set(
                self.user_item_matrix.columns[
                    self.user_item_matrix.loc[user_id] > 0
                ]
            )

        print(f"🔍 Generating item-based recommendations for User {user_id}...")

        predictions = []
        candidate_movies = [m for m in self.movie_ids if m not in rated_ids]

        for movie_id in candidate_movies:
            pred = self.predict_rating(user_id, movie_id)
            if pred is not None:
                predictions.append({
                    "movie_id": movie_id,
                    "predicted_rating": round(pred, 3)
                })

        if not predictions:
            return pd.DataFrame(columns=["movie_id", "predicted_rating"])

        return (
            pd.DataFrame(predictions)
            .sort_values("predicted_rating", ascending=False)
            .head(n)
            .reset_index(drop=True)
        )


# ─────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────

def evaluate_knn(model, test_df, sample_size=1000):
    """
    Computes RMSE and MAE on held-out test ratings.

    RMSE (Root Mean Square Error):
    - Penalizes large errors heavily (squared)
    - "On average, my predictions are off by X stars"
    - Good RMSE for ml-100k: < 1.0

    MAE (Mean Absolute Error):
    - Average absolute difference between predicted and actual
    - More interpretable than RMSE
    - Good MAE for ml-100k: < 0.80

    Args:
        model: Fitted UserBasedKNN or ItemBasedKNN
        test_df: Held-out ratings DataFrame
        sample_size: Evaluate on a random sample (full eval is slow for k-NN)

    Returns:
        dict with rmse, mae, coverage
    """
    sample = test_df.sample(min(sample_size, len(test_df)), random_state=42)

    actuals = []
    predictions = []
    unpredicted = 0

    print(f"📊 Evaluating on {len(sample)} test ratings...")

    for _, row in sample.iterrows():
        pred = model.predict_rating(int(row["user_id"]), int(row["movie_id"]))
        if pred is not None:
            actuals.append(row["rating"])
            predictions.append(pred)
        else:
            unpredicted += 1

    actuals = np.array(actuals)
    predictions = np.array(predictions)

    rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
    mae = np.mean(np.abs(actuals - predictions))
    coverage = len(predictions) / len(sample)

    print(f"\n{'─'*40}")
    print(f"  📈 Evaluation Results [{model.__class__.__name__}]")
    print(f"{'─'*40}")
    print(f"  RMSE:     {rmse:.4f}  (target: < 1.0)")
    print(f"  MAE:      {mae:.4f}  (target: < 0.80)")
    print(f"  Coverage: {coverage:.1%}  (% of test ratings predictable)")
    print(f"  Unpredicted: {unpredicted} ratings (cold start)")
    print(f"{'─'*40}\n")

    return {"rmse": rmse, "mae": mae, "coverage": coverage}


# ─────────────────────────────────────────────
# QUICK TEST — run directly
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))

    from data_loader import load_all
    from utils import enrich_recommendations, print_recommendations

    # Load data
    ratings, movies, users, matrix, train, test = load_all()

    # ── USER-BASED ──
    print("\n" + "═"*60)
    print("  USER-BASED k-NN")
    print("═"*60)

    user_knn = UserBasedKNN(k=20, min_common_ratings=3)
    user_knn.fit(train)

    # Recommend for user 1
    recs = user_knn.recommend(user_id=1, n=10, ratings_df=train)
    enriched = enrich_recommendations(recs, movies)
    print_recommendations(enriched.reset_index(drop=True), user_id=1, method="User-Based k-NN")

    # Evaluate
    user_eval = evaluate_knn(user_knn, test, sample_size=500)

    # ── ITEM-BASED ──
    print("\n" + "═"*60)
    print("  ITEM-BASED k-NN")
    print("═"*60)

    item_knn = ItemBasedKNN(k=20)
    item_knn.fit(train)

    recs_item = item_knn.recommend(user_id=1, n=10, ratings_df=train)
    enriched_item = enrich_recommendations(recs_item, movies)
    print_recommendations(enriched_item.reset_index(drop=True), user_id=1, method="Item-Based k-NN")

    item_eval = evaluate_knn(item_knn, test, sample_size=500)

    # ── COMPARE ──
    print("\n── Model Comparison ──")
    print(f"{'Model':<20} {'RMSE':>8} {'MAE':>8} {'Coverage':>10}")
    print("─" * 50)
    print(f"{'UserBasedKNN':<20} {user_eval['rmse']:>8.4f} "
          f"{user_eval['mae']:>8.4f} {user_eval['coverage']:>10.1%}")
    print(f"{'ItemBasedKNN':<20} {item_eval['rmse']:>8.4f} "
          f"{item_eval['mae']:>8.4f} {item_eval['coverage']:>10.1%}")