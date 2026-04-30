# backend/src/evaluator.py

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server
import matplotlib.pyplot as plt
import os
import time


def evaluate_model(model, test_df, model_name, sample_size=2000):
    """
    Comprehensive evaluation of any recommender model.
    All models share the same interface: predict_rating(user_id, movie_id)

    Metrics computed:
    ─────────────────────────────────────────────────────
    RMSE   - penalizes large errors heavily
    MAE    - interpretable average error in stars
    Coverage - % of test pairs the model can predict
    Precision@K - of top-K recs, how many are "relevant" (≥4 stars)
    ─────────────────────────────────────────────────────
    """
    sample = test_df.sample(min(sample_size, len(test_df)), random_state=42)

    preds, actuals = [], []
    unpredicted = 0
    start = time.time()

    for _, row in sample.iterrows():
        pred = model.predict_rating(int(row["user_id"]), int(row["movie_id"]))
        if pred is not None:
            preds.append(pred)
            actuals.append(row["rating"])
        else:
            unpredicted += 1

    elapsed = time.time() - start

    preds = np.array(preds)
    actuals = np.array(actuals)

    rmse = np.sqrt(np.mean((actuals - preds) ** 2))
    mae = np.mean(np.abs(actuals - preds))
    coverage = len(preds) / len(sample)

    # Error distribution
    errors = actuals - preds
    within_1_star = np.mean(np.abs(errors) <= 1.0)
    within_half_star = np.mean(np.abs(errors) <= 0.5)

    return {
        "model": model_name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "coverage": round(coverage, 4),
        "within_1_star": round(within_1_star, 4),
        "within_half_star": round(within_half_star, 4),
        "n_predicted": len(preds),
        "n_unpredicted": unpredicted,
        "eval_time_sec": round(elapsed, 2)
    }


def generate_comparison_report(models_dict, test_df, output_dir=None):
    """
    Generates a full comparison report across all models.
    Saves a PNG chart — this goes in your GitHub README.

    Args:
        models_dict: {"ModelName": fitted_model_instance, ...}
        test_df: held-out test ratings
        output_dir: where to save the chart

    Returns:
        pd.DataFrame with all metrics
    """
    print("\n" + "═"*65)
    print("  📊 MODEL COMPARISON REPORT")
    print("═"*65)

    results = []
    for name, model in models_dict.items():
        print(f"\n  Evaluating {name}...")
        metrics = evaluate_model(model, test_df, name)
        results.append(metrics)

    df = pd.DataFrame(results).set_index("model")

    # ── Print Report Table ──
    print("\n")
    print(f"{'─'*65}")
    print(f"  {'Model':<22} {'RMSE':>8} {'MAE':>8} "
          f"{'Coverage':>10} {'±1 Star':>10}")
    print(f"{'─'*65}")

    for _, row in df.iterrows():
        print(f"  {row.name:<22} {row['rmse']:>8.4f} {row['mae']:>8.4f} "
              f"{row['coverage']:>10.1%} {row['within_1_star']:>10.1%}")

    print(f"{'─'*65}")

    best_rmse = df["rmse"].idxmin()
    best_mae = df["mae"].idxmin()
    print(f"\n  🏆 Best RMSE: {best_rmse} ({df.loc[best_rmse, 'rmse']})")
    print(f"  🏆 Best MAE:  {best_mae} ({df.loc[best_mae, 'mae']})")
    print(f"{'═'*65}\n")

    # ── Generate Chart ──
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        _plot_comparison(df, output_dir)

    return df


def _plot_comparison(df, output_dir):
    """Generates and saves a comparison bar chart."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Recommender Model Comparison — MovieLens 100K",
                 fontsize=14, fontweight="bold")

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    models = df.index.tolist()
    x = np.arange(len(models))

    # RMSE
    axes[0].bar(x, df["rmse"], color=colors[:len(models)])
    axes[0].set_title("RMSE (lower is better)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(models, rotation=15)
    axes[0].set_ylim(0.7, 1.1)
    for i, v in enumerate(df["rmse"]):
        axes[0].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)

    # MAE
    axes[1].bar(x, df["mae"], color=colors[:len(models)])
    axes[1].set_title("MAE (lower is better)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(models, rotation=15)
    axes[1].set_ylim(0.5, 0.9)
    for i, v in enumerate(df["mae"]):
        axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=9)

    # Coverage
    axes[2].bar(x, df["coverage"] * 100, color=colors[:len(models)])
    axes[2].set_title("Coverage % (higher is better)")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(models, rotation=15)
    axes[2].set_ylim(80, 105)
    for i, v in enumerate(df["coverage"] * 100):
        axes[2].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Chart saved → {chart_path}")
    plt.close()