# backend/src/run_day3.py
# Run this to see the full Day 3 output

import sys, os
sys.path.append(os.path.dirname(__file__))

from data_loader import load_all
from recommender_knn import UserBasedKNN, ItemBasedKNN
from recommender_svd import SVDRecommender, tune_svd
from evaluator import generate_comparison_report
from utils import enrich_recommendations, print_recommendations

# ── Load Data ──
ratings, movies, users, matrix, train, test = load_all()

# ── Train All Three Models ──
print("\n" + "═"*60)
print("  TRAINING ALL MODELS")
print("═"*60)

user_knn = UserBasedKNN(k=20).fit(train)
item_knn = ItemBasedKNN(k=20).fit(train)
svd = SVDRecommender(n_factors=100, n_epochs=20).fit(train)

# ── Full Comparison Report ──
report = generate_comparison_report(
    models_dict={
        "UserBased-KNN": user_knn,
        "ItemBased-KNN": item_knn,
        "SVD": svd
    },
    test_df=test,
    output_dir="backend/notebooks"
)

# ── SVD Recommendations for User 1 ──
recs = svd.recommend(user_id=1, n=10, ratings_df=train)
enriched = enrich_recommendations(recs, movies)
print_recommendations(enriched.reset_index(drop=True),
                      user_id=1, method="SVD")

# ── Bonus: Similar Movies ──
print("\n── Movies similar to 'Star Wars (1977)' ──")
star_wars_id = movies[movies["title"].str.contains("Star Wars")
                      ]["movie_id"].values[0]
similar = svd.get_similar_movies(star_wars_id, n=5, movies_df=movies)
print(similar[["title", "year", "similarity"]].to_string(index=False))

# ── Optional: Hyperparameter Tuning ──
# Uncomment to run (takes ~5 mins):
# tune_svd(train, test, factor_options=[20, 50, 100, 150])

# ── Save best model for Day 4 ──
os.makedirs("backend/models", exist_ok=True)
svd.save("backend/models/svd_model.pkl")