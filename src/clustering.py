"""
Clustering module.

Design decisions worth calling out explicitly:

1. SCALING: RFM-style features are heavily right-skewed (a handful of
   customers spend/order far more than the median). K-Means uses Euclidean
   distance, so an un-transformed 'monetary_total' feature (range: tens to
   thousands) would dominate a 0-1 bounded feature like 'category_diversity'.
   We log-transform skewed features, then standard-scale everything -- this
   is the standard, defensible preprocessing for K-Means on retail data.

2. K SELECTION: we never just "pick k=4 because it looked right". We compute
   both the elbow (inertia) curve AND silhouette scores across a range of k,
   and choose the k that maximises silhouette (a bounded, interpretable
   metric of how well-separated clusters are) rather than eyeballing an
   elbow, which is subjective.

3. VALIDATION AGAINST A SECOND ALGORITHM: shipping a single clustering
   result with no comparison is a common student-project mistake. We fit
   Gaussian Mixture Models at the same k and report agreement (Adjusted Rand
   Index) between the two -- if two different algorithms broadly agree on
   the grouping, that's evidence the structure is real rather than an
   artifact of K-Means' spherical-cluster assumption.
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.decomposition import PCA

from config import (
    CLUSTERING_FEATURES, RANDOM_STATE, K_SEARCH_RANGE, KMEANS_N_INIT,
    MIN_BUSINESS_K, MAX_BUSINESS_K,
)

logger = logging.getLogger(__name__)

# Features that are heavily right-skewed and benefit from log1p before scaling
SKEWED_FEATURES = [
    "monetary_total", "avg_order_value", "recency_days",
    "purchase_regularity", "tenure_days",
]


def prepare_feature_matrix(features_df: pd.DataFrame):
    """Log-transforms skewed features, then standardizes all clustering
    features. Returns the scaled numpy array, the fitted scaler (needed to
    transform new customers later), and the transformed dataframe for
    inspection."""
    X = features_df[CLUSTERING_FEATURES].copy()

    for col in SKEWED_FEATURES:
        if col in X.columns:
            X[col] = np.log1p(X[col])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, X


def select_optimal_k(X_scaled, k_range=K_SEARCH_RANGE):
    """Fits K-Means across a range of k and returns a dataframe of
    inertia + silhouette score per k, plus the k that maximises silhouette."""
    results = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        logger.info(f"k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}")

    results_df = pd.DataFrame(results)

    global_best_k = int(results_df.loc[results_df["silhouette"].idxmax(), "k"])

    business_mask = (results_df["k"] >= MIN_BUSINESS_K) & (results_df["k"] <= MAX_BUSINESS_K)
    business_df = results_df[business_mask]
    business_best_k = int(business_df.loc[business_df["silhouette"].idxmax(), "k"])

    if global_best_k != business_best_k:
        logger.warning(
            f"Statistically optimal k={global_best_k} (silhouette="
            f"{results_df.loc[results_df['k']==global_best_k,'silhouette'].values[0]:.4f}) "
            f"falls outside the business-usable range [{MIN_BUSINESS_K},{MAX_BUSINESS_K}]. "
            f"Selecting k={business_best_k} instead (silhouette="
            f"{business_df.loc[business_df['k']==business_best_k,'silhouette'].values[0]:.4f}) "
            f"as it yields actionable, differentiable segments for marketing."
        )

    logger.info(f"Selected k={business_best_k} for final segmentation")
    return results_df, business_best_k


def fit_kmeans(X_scaled, k):
    km = KMeans(n_clusters=k, n_init=KMEANS_N_INIT, random_state=RANDOM_STATE)
    labels = km.fit_predict(X_scaled)
    return km, labels


def fit_gmm(X_scaled, k):
    gmm = GaussianMixture(n_components=k, random_state=RANDOM_STATE, n_init=5)
    labels = gmm.fit_predict(X_scaled)
    return gmm, labels


def validate_clustering(X_scaled, kmeans_labels, k):
    """Cross-checks K-Means against GMM at the same k. A high Adjusted Rand
    Index (closer to 1) means the two algorithms broadly agree on customer
    groupings despite making different structural assumptions -- evidence
    the segments reflect real structure, not a K-Means artifact."""
    _, gmm_labels = fit_gmm(X_scaled, k)
    ari = adjusted_rand_score(kmeans_labels, gmm_labels)
    sil_kmeans = silhouette_score(X_scaled, kmeans_labels)
    sil_gmm = silhouette_score(X_scaled, gmm_labels)
    logger.info(f"Validation @ k={k}: KMeans silhouette={sil_kmeans:.4f}, "
                f"GMM silhouette={sil_gmm:.4f}, Adjusted Rand Index={ari:.4f}")
    return {"kmeans_silhouette": sil_kmeans, "gmm_silhouette": sil_gmm, "ari": ari}


def project_to_2d(X_scaled):
    """PCA projection purely for visualization -- clustering itself happens
    in the full feature space, not on these 2 components."""
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_
    logger.info(f"PCA 2D projection explains {explained.sum()*100:.1f}% of variance "
                f"(PC1={explained[0]*100:.1f}%, PC2={explained[1]*100:.1f}%)")
    return coords, pca


if __name__ == "__main__":
    from config import CUSTOMER_FEATURES_FILE, CLUSTER_ASSIGNMENTS_FILE

    features_df = pd.read_csv(CUSTOMER_FEATURES_FILE)
    X_scaled, scaler, X_transformed = prepare_feature_matrix(features_df)

    k_results, best_k = select_optimal_k(X_scaled)
    km_model, labels = fit_kmeans(X_scaled, best_k)
    validation = validate_clustering(X_scaled, labels, best_k)
    coords, pca_model = project_to_2d(X_scaled)

    out = features_df.copy()
    out["cluster"] = labels
    out["pca_x"] = coords[:, 0]
    out["pca_y"] = coords[:, 1]
    CLUSTER_ASSIGNMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLUSTER_ASSIGNMENTS_FILE, index=False)

    print(k_results)
    print(f"\nBest k: {best_k}")
    print(f"Validation: {validation}")
    print(f"Cluster sizes:\n{out['cluster'].value_counts().sort_index()}")
