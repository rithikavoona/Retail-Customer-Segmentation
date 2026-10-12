"""
Generates every static visualization needed for the project report:
  1. K-selection diagnostic (elbow + silhouette side by side)
  2. PCA 2D scatter of customers colored by segment
  3. Segment size distribution
  4. Per-feature boxplots by segment (distribution, not just the mean)
  5. Radar/spider chart comparing segment profiles on normalized features
  6. CLV contribution by segment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import FIGURES_DIR

sns.set_theme(style="whitegrid", palette="deep")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_k_selection(k_results: pd.DataFrame, chosen_k: int, save_name="01_k_selection.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].plot(k_results["k"], k_results["inertia"], marker="o", color="#2C6E9B")
    axes[0].axvline(chosen_k, color="#D9534F", linestyle="--", alpha=0.7, label=f"Chosen k={chosen_k}")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
    axes[0].set_title("Elbow Method")
    axes[0].legend()

    axes[1].plot(k_results["k"], k_results["silhouette"], marker="o", color="#3E8E57")
    axes[1].axvline(chosen_k, color="#D9534F", linestyle="--", alpha=0.7, label=f"Chosen k={chosen_k}")
    axes[1].set_xlabel("Number of clusters (k)")
    axes[1].set_ylabel("Silhouette Score")
    axes[1].set_title("Silhouette Analysis")
    axes[1].legend()

    fig.suptitle("Optimal Cluster Count Selection", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, dpi=150)
    plt.close(fig)


def plot_pca_scatter(cluster_df: pd.DataFrame, segment_names: dict, save_name="02_pca_clusters.png"):
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = cluster_df["cluster"].map(segment_names)
    sns.scatterplot(data=cluster_df, x="pca_x", y="pca_y", hue=labels,
                     palette="deep", s=45, alpha=0.75, ax=ax)
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")
    ax.set_title("Customer Segments (PCA 2D Projection)", fontweight="bold")
    ax.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, dpi=150)
    plt.close(fig)


def plot_segment_sizes(profile: pd.DataFrame, save_name="03_segment_sizes.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = profile.sort_values("customer_count", ascending=False)
    bars = ax.bar(data["segment_name"], data["customer_count"], color=sns.color_palette("deep", len(data)))
    for bar, pct in zip(bars, data["pct_of_customer_base"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                f"{pct}%", ha="center", fontsize=9)
    ax.set_ylabel("Number of Customers")
    ax.set_title("Customer Segment Sizes", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, dpi=150)
    plt.close(fig)


def plot_feature_boxplots(cluster_df: pd.DataFrame, segment_names: dict, save_name="04_feature_distributions.png"):
    plot_df = cluster_df.copy()
    plot_df["segment"] = plot_df["cluster"].map(segment_names)
    key_features = ["recency_days", "frequency", "monetary_total", "avg_order_value"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feat in zip(axes.flat, key_features):
        sns.boxplot(data=plot_df, x="segment", y=feat, hue="segment", ax=ax, palette="deep", legend=False)
        ax.set_title(feat.replace("_", " ").title())
        ax.set_xlabel("")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.suptitle("Feature Distributions by Segment", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, dpi=150)
    plt.close(fig)


def plot_radar_chart(profile: pd.DataFrame, save_name="05_segment_radar.png"):
    """Compares segments across normalized features on a radar/spider chart --
    the standard visual for comparing multi-dimensional cluster profiles at a
    glance, widely used in customer-segmentation deliverables."""
    radar_features = ["recency_days", "frequency", "monetary_total",
                       "avg_basket_size", "category_diversity"]
    norm = profile[radar_features].copy()
    # Invert recency so that "higher is better" holds for every axis
    norm["recency_days"] = -norm["recency_days"]
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-9)

    angles = np.linspace(0, 2*np.pi, len(radar_features), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    colors = sns.color_palette("deep", len(norm))
    for (idx, row), color in zip(norm.iterrows(), colors):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, linewidth=2, label=profile.loc[idx, "segment_name"], color=color)
        ax.fill(angles, values, alpha=0.08, color=color)

    labels = ["Recency\n(inverted)", "Frequency", "Monetary", "Basket Size", "Category\nDiversity"]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Segment Profile Comparison (normalized)", fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1))
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, dpi=150)
    plt.close(fig)


def plot_clv_contribution(profile: pd.DataFrame, save_name="06_clv_by_segment.png"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    data = profile.sort_values("total_clv", ascending=False)

    axes[0].bar(data["segment_name"], data["total_clv"], color=sns.color_palette("deep", len(data)))
    axes[0].set_ylabel(f"Estimated Total {12}-Month CLV")
    axes[0].set_title("Total Projected Value by Segment", fontweight="bold")
    plt.setp(axes[0].get_xticklabels(), rotation=20, ha="right")

    axes[1].bar(data["segment_name"], data["clv_estimate"], color=sns.color_palette("deep", len(data)))
    axes[1].set_ylabel("Estimated CLV per Customer")
    axes[1].set_title("Per-Customer Value by Segment", fontweight="bold")
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, dpi=150)
    plt.close(fig)


def plot_som_map(som, X_scaled, cluster_labels, segment_names, save_name="07_som_map.png"):
    """
    Visualizes the trained SOM as two side-by-side panels:
      (left)  U-Matrix -- unsupervised view of map structure. Bright
              (high-distance) regions mark natural boundaries between
              groups of similar neurons, found with NO knowledge of the
              K-Means labels.
      (right) The same grid, but each customer's Best Matching Unit is
              plotted and colored by its K-Means segment. If K-Means
              segments occupy visually distinct, coherent regions of the
              SOM grid, that's independent supporting evidence the
              segments reflect real structure -- two very different
              algorithms (competitive learning vs. centroid-based
              partitioning) agreeing on the shape of the customer base.
    """
    u_mat = som.u_matrix()
    bmu_coords = som.transform(X_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    im = axes[0].imshow(u_mat, cmap="bone", origin="lower")
    axes[0].set_title("SOM U-Matrix\n(unsupervised map structure)", fontweight="bold")
    axes[0].set_xlabel("Grid column")
    axes[0].set_ylabel("Grid row")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Avg. distance to neighbours")

    axes[1].imshow(u_mat, cmap="Greys", origin="lower", alpha=0.35)
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, size=bmu_coords.shape)
    labels = pd.Series(cluster_labels).map(segment_names)
    for name in labels.unique():
        mask = labels == name
        pts = bmu_coords[mask.values] + jitter[mask.values]
        axes[1].scatter(pts[:, 1], pts[:, 0], label=name, s=22, alpha=0.75)
    axes[1].set_title("Customer Positions on SOM,\ncolored by K-Means Segment", fontweight="bold")
    axes[1].set_xlabel("Grid column")
    axes[1].set_ylabel("Grid row")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

    fig.suptitle("Self-Organizing Map: Independent Cross-Check of Segment Structure",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / save_name, dpi=150)
    plt.close(fig)


def generate_all_figures(k_results, chosen_k, cluster_df, profile, segment_names):
    plot_k_selection(k_results, chosen_k)
    plot_pca_scatter(cluster_df, segment_names)
    plot_segment_sizes(profile)
    plot_feature_boxplots(cluster_df, segment_names)
    plot_radar_chart(profile)
    plot_clv_contribution(profile)
    print(f"All figures saved to {FIGURES_DIR}")
