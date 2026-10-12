"""
End-to-end pipeline orchestration.

Run with:  python3 main.py

This is intentionally a thin script: all real logic lives in the
single-purpose modules (data_loader, preprocessing, feature_engineering,
clustering, profiling, visualization). main.py just sequences them and
handles I/O -- this separation is what makes each stage independently
testable and reusable (e.g. the dashboard reuses profiling.py directly).
"""

import json
import logging
import numpy as np
import pandas as pd

from config import (
    RAW_TRANSACTIONS_FILE, CLEANED_TRANSACTIONS_FILE, CUSTOMER_FEATURES_FILE,
    CLUSTER_ASSIGNMENTS_FILE, SEGMENT_SUMMARY_JSON, COL_INVOICE_DATE,
)
from data_loader import load_raw_transactions
from preprocessing import clean_pipeline
from feature_engineering import build_customer_features
from clustering import prepare_feature_matrix, select_optimal_k, fit_kmeans, validate_clustering, project_to_2d
from profiling import build_segment_profiles, name_segments, get_playbook
from visualization import generate_all_figures, plot_som_map
from som import SimpleSOM

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline():
    logger.info("=" * 70)
    logger.info("RETAIL CUSTOMER SEGMENTATION PIPELINE")
    logger.info("=" * 70)

    # 1. Load
    raw_df = load_raw_transactions(RAW_TRANSACTIONS_FILE)

    # 2. Clean
    clean_df = clean_pipeline(raw_df)
    CLEANED_TRANSACTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(CLEANED_TRANSACTIONS_FILE, index=False)

    # 3. Feature engineering
    features_df = build_customer_features(clean_df)
    features_df.to_csv(CUSTOMER_FEATURES_FILE, index=False)

    # 4. Clustering
    X_scaled, scaler, X_transformed = prepare_feature_matrix(features_df)
    k_results, best_k = select_optimal_k(X_scaled)
    km_model, labels = fit_kmeans(X_scaled, best_k)
    validation = validate_clustering(X_scaled, labels, best_k)
    coords, pca_model = project_to_2d(X_scaled)

    cluster_df = features_df.copy()
    cluster_df["cluster"] = labels
    cluster_df["pca_x"] = coords[:, 0]
    cluster_df["pca_y"] = coords[:, 1]
    cluster_df.to_csv(CLUSTER_ASSIGNMENTS_FILE, index=False)

    # 5. Business profiling
    profile = build_segment_profiles(cluster_df)
    segment_names = name_segments(profile)
    profile["segment_name"] = profile.index.map(segment_names)

    # 6. Visualizations
    generate_all_figures(k_results, best_k, cluster_df, profile, segment_names)

    # 6b. SOM (syllabus Module 4) -- independent, non-linear cross-check of
    # the K-Means segmentation, trained on the same scaled feature matrix.
    logger.info("Training Self-Organizing Map (bonus cross-validation view)...")
    grid_size = max(5, int(np.ceil(np.sqrt(5 * np.sqrt(len(X_scaled))))))  # common SOM sizing heuristic
    som = SimpleSOM(grid_rows=grid_size, grid_cols=grid_size, input_dim=X_scaled.shape[1])
    som.train(X_scaled, n_iterations=3000, verbose_every=1000)
    plot_som_map(som, X_scaled, labels, segment_names)

    # 7. Export dashboard data (self-contained JSON consumed by dashboard/index.html)
    export_dashboard_json(profile, segment_names, validation, best_k, cluster_df, clean_df)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    print_summary(profile, validation, best_k)

    return {
        "clean_df": clean_df, "features_df": features_df, "cluster_df": cluster_df,
        "profile": profile, "segment_names": segment_names, "validation": validation,
        "best_k": best_k, "k_results": k_results,
    }


def export_dashboard_json(profile, segment_names, validation, best_k, cluster_df, clean_df):
    segments = []
    for cid, row in profile.iterrows():
        name = row["segment_name"]
        pb = get_playbook(name)
        segments.append({
            "cluster_id": int(cid),
            "name": name,
            "customer_count": int(row["customer_count"]),
            "pct_of_base": float(row["pct_of_customer_base"]),
            "recency_days": float(row["recency_days"]),
            "frequency": float(row["frequency"]),
            "monetary_total": float(row["monetary_total"]),
            "avg_order_value": float(row["avg_order_value"]),
            "avg_basket_size": float(row["avg_basket_size"]),
            "category_diversity": float(row["category_diversity"]),
            "clv_per_customer": float(row["clv_estimate"]),
            "total_clv": float(row["total_clv"]),
            "strategy": pb["strategy"],
            "rationale": pb["rationale"],
        })

    summary = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "total_customers": int(len(cluster_df)),
        "total_transactions_analyzed": int(len(clean_df)),
        "date_range": {
            "start": str(clean_df[COL_INVOICE_DATE].min().date()),
            "end": str(clean_df[COL_INVOICE_DATE].max().date()),
        },
        "chosen_k": int(best_k),
        "validation": {k: round(float(v), 4) for k, v in validation.items()},
        "segments": sorted(segments, key=lambda s: -s["total_clv"]),
    }

    SEGMENT_SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SEGMENT_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Dashboard data exported to {SEGMENT_SUMMARY_JSON}")


def print_summary(profile, validation, best_k):
    print("\n" + "=" * 70)
    print("SEGMENTATION SUMMARY")
    print("=" * 70)
    print(f"Chosen k = {best_k}")
    print(f"K-Means silhouette: {validation['kmeans_silhouette']:.4f}")
    print(f"GMM cross-validation silhouette: {validation['gmm_silhouette']:.4f}")
    print(f"Adjusted Rand Index (KMeans vs GMM agreement): {validation['ari']:.4f}")
    print("\nSegments:")
    cols = ["segment_name", "customer_count", "pct_of_customer_base",
            "recency_days", "frequency", "monetary_total", "clv_estimate", "total_clv"]
    print(profile[cols].to_string())


if __name__ == "__main__":
    run_pipeline()
