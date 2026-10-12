"""
FastAPI Server for Retail Customer Segmentation Web Application.
Supports multiple datasets, multiple ML clustering algorithms (K-Means, GMM, Agglomerative, DBSCAN, SOM),
customer profiling, interactive inference simulator, and ROI calculation.
"""

import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score, adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import CLV_HORIZON_MONTHS, ASSUMED_GROSS_MARGIN
from data_loader import load_raw_transactions
from preprocessing import clean_pipeline
from feature_engineering import build_customer_features
from profiling import compute_clv
from som import SimpleSOM

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("retail_segmentation_api")

app = FastAPI(title="Retail Customer Segmentation Intelligence Platform", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
STATIC_DIR = Path(__file__).resolve().parent / "static"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)

# Datasets catalog
DATASETS_INFO = {
    "online_retail_real": {
        "id": "online_retail_real",
        "name": "UCI Online Retail (Official Benchmark - 541K Records)",
        "description": "Real transaction data from a UK-based non-store online retail enterprise (2010–2011). 541,909 raw records, 4,334 active customers after rigorous cleaning.",
        "type": "transactional",
        "raw_file": "online_retail_real.csv",
        "features_file": "features_online_retail_real.csv",
        "transaction_count": 541909,
        "clean_count": 396195,
        "customer_count": 4334,
        "currency": "£",
        "default_k": 4,
    },
    "online_retail_ii": {
        "id": "online_retail_ii",
        "name": "Online Retail II (Curated Sample - 46K Records)",
        "description": "Representative 1,493-customer transaction sample replicating real-world e-commerce purchase distributions, cancellations, and order rhythms.",
        "type": "transactional",
        "raw_file": "online_retail_ii.csv",
        "features_file": "features_online_retail_ii.csv",
        "transaction_count": 46084,
        "clean_count": 41283,
        "customer_count": 1493,
        "currency": "£",
        "default_k": 3,
    },
    "mall_customers": {
        "id": "mall_customers",
        "name": "Mall Customer Segmentation Dataset (200 Records)",
        "description": "Classic retail store customer dataset featuring Annual Income ($k), Spending Score (1-100), and Age attributes for behavioral clustering.",
        "type": "customer_summary",
        "raw_file": "mall_customers.csv",
        "features_file": "features_mall_customers.csv",
        "transaction_count": 200,
        "clean_count": 200,
        "customer_count": 200,
        "currency": "$",
        "default_k": 5,
    },
}

DEFAULT_RFM_FEATURES = [
    "recency_days",
    "frequency",
    "monetary_total",
    "avg_order_value",
    "avg_basket_size",
    "purchase_regularity",
    "category_diversity",
    "tenure_days",
]

SKEWED_FEATURES = [
    "monetary_total",
    "avg_order_value",
    "recency_days",
    "purchase_regularity",
    "tenure_days",
]

# Memory cache for active clustering models
CURRENT_STATE = {
    "dataset_id": "online_retail_real",
    "features_df": None,
    "scaled_X": None,
    "scaler": None,
    "model": None,
    "labels": None,
    "algorithm": "kmeans",
    "k": 4,
    "cluster_names": {},
    "feature_cols": DEFAULT_RFM_FEATURES,
}


def load_dataset_features(dataset_id: str) -> pd.DataFrame:
    """Loads features dataframe for given dataset_id."""
    if dataset_id not in DATASETS_INFO:
        # Check if custom uploaded
        custom_path = DATA_PROCESSED_DIR / f"features_{dataset_id}.csv"
        if custom_path.exists():
            return pd.read_csv(custom_path)
        raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found.")

    info = DATASETS_INFO[dataset_id]
    features_path = DATA_PROCESSED_DIR / info["features_file"]
    if not features_path.exists():
        # Fallback to customer_features.csv
        features_path = DATA_PROCESSED_DIR / "customer_features.csv"
    return pd.read_csv(features_path)


def assign_persona_names(profile: pd.DataFrame, is_mall: bool = False) -> Dict[int, str]:
    """Dynamically assigns descriptive business persona names based on cluster metrics."""
    if is_mall:
        # Mall customers naming based on Income vs Spending Score
        names = {}
        inc_med = profile["annual_income"].median()
        spend_med = profile["spending_score"].median()
        for cid, row in profile.iterrows():
            inc = row["annual_income"]
            spend = row["spending_score"]
            if inc >= inc_med and spend >= spend_med:
                names[cid] = "Affluent Big Spenders (VIP)"
            elif inc < inc_med and spend >= spend_med:
                names[cid] = "Carefree Spenders (Youth/Fashion)"
            elif inc >= inc_med and spend < spend_med:
                names[cid] = "Cautious Affluents (Savers)"
            elif inc < inc_med and spend < spend_med:
                names[cid] = "Frugal / Budget Shoppers"
            else:
                names[cid] = "Balanced Regulars"
        return names

    # RFM Persona assignment
    def _minmax(s):
        lo, hi = s.min(), s.max()
        return (s - lo) / (hi - lo) if (hi - lo) > 1e-9 else pd.Series(0.5, index=s.index)

    r_score = 1.0 - _minmax(profile["recency_days"])  # low recency = high score
    f_score = _minmax(profile["frequency"])
    m_score = _minmax(profile["monetary_total"])
    rfm_score = (r_score + f_score + m_score) / 3.0

    overall_median_aov = profile["avg_order_value"].median() if "avg_order_value" in profile else 0

    names = {}
    assigned = []
    # Rank by rfm_score
    sorted_cids = rfm_score.sort_values(ascending=False).index.tolist()

    for idx, cid in enumerate(sorted_cids):
        row = profile.loc[cid]
        rec = row["recency_days"]
        freq = row["frequency"]
        mon = row["monetary_total"]
        score = rfm_score.loc[cid]
        aov = row.get("avg_order_value", mon / max(freq, 1))

        if idx == 0 and score >= 0.5:
            name = "Champions / High-Value VIPs"
        elif rec < profile["recency_days"].quantile(0.4) and freq >= profile["frequency"].median():
            name = "Potential Loyalists / Steady Regulars"
        elif rec < profile["recency_days"].quantile(0.4) and freq <= 2:
            name = "Promising New Buyers"
        elif rec >= profile["recency_days"].quantile(0.6) and (aov > overall_median_aov or mon > profile["monetary_total"].median()):
            name = "At-Risk High-Value Customers"
        elif rec >= profile["recency_days"].quantile(0.7):
            name = "Hibernating / Churned Customers"
        elif aov < overall_median_aov and freq > profile["frequency"].quantile(0.6):
            name = "Bargain Hunters / Discount Seekers"
        else:
            name = "Occasional / Developing Shoppers"

        # Avoid exact duplicate names by adding suffix if needed
        base_name = name
        counter = 2
        while name in assigned:
            name = f"{base_name} ({counter})"
            counter += 1
        assigned.append(name)
        names[cid] = name

    return names


EXTENDED_PLAYBOOK = {
    "Champions / High-Value VIPs": {
        "strategy": "VIP Tiers, Exclusive Perks & Loyalty Rewards",
        "rationale": "Customers with top-tier recency, frequent transactions, and peak monetary spend. They drive the lion's share of profitability. Retention is exponentially cheaper than re-acquisition.",
        "key_channels": ["Dedicated Account Manager", "VIP Concierge Email", "Private SMS Previews"],
        "discount_sensitivity": "Very Low — Prioritize recognition, early access, and premier packaging over price discounts.",
        "churn_risk_level": "Low (but critical impact if lost)",
        "action_item": "Enroll in Platinum VIP tier, provide early drop access, offer surprise appreciation gifts, and solicit NPS feedback."
    },
    "Potential Loyalists / Steady Regulars": {
        "strategy": "Cross-Selling, Personalized Recommendations & Habit Building",
        "rationale": "Engaged shoppers who buy consistently with good recency. Upselling basket additions and cross-category discovery can elevate them into Champions.",
        "key_channels": ["Personalized Email", "In-App Product Triggers", "Push Notifications"],
        "discount_sensitivity": "Medium — Bundle discounts and threshold rewards (e.g., 'Spend £75, Get Free Gift') work exceptionally well.",
        "churn_risk_level": "Low to Moderate",
        "action_item": "Trigger recommendation engines tailored to their category affinities and introduce point-multiplier loyalty challenges."
    },
    "Promising New Buyers": {
        "strategy": "Warm Onboarding, Brand Immersion & Second-Order Incentives",
        "rationale": "High recency but low transaction count (1-2 orders). They are evaluating your brand experience; getting a second purchase within 30 days increases lifetime retention by 300%.",
        "key_channels": ["Welcome Email Drip", "SMS First-Order Follow-up", "Retargeting Ads"],
        "discount_sensitivity": "Moderate — Welcome back discount with an expiration countdown.",
        "churn_risk_level": "High drop-off risk without proactive onboarding",
        "action_item": "Send onboarding sequence with care instructions, community highlights, and a 14-day expiry coupon on their next purchase."
    },
    "At-Risk High-Value Customers": {
        "strategy": "Proactive Win-Back, Direct Outreach & High-Impact Incentives",
        "rationale": "Previously substantial spenders who have become dormant (high recency days). These customers represent high recoverable equity and ROI if reactivated.",
        "key_channels": ["Executive Win-Back Email", "Direct SMS", "Targeted Paid Retargeting"],
        "discount_sensitivity": "High — Significant reactivation incentive or personalized gift is justified by their high historical spend.",
        "churn_risk_level": "High — Approaching permanent dormancy",
        "action_item": "'We Miss You' campaign featuring past favorite product lines, a 20% reactivation offer, and feedback inquiry on service quality."
    },
    "Hibernating / Churned Customers": {
        "strategy": "Low-Cost Automated Reactivation & Marketing Hygiene",
        "rationale": "Long inactivity and low historical transaction value. Expensive direct marketing will generate negative ROI; rely on automated mass channels or unsubscribe prune.",
        "key_channels": ["Quarterly Clearance Newsletter", "Automated Win-Back Sequences"],
        "discount_sensitivity": "High — Steep discounts (clearance/flash sales) only.",
        "churn_risk_level": "Critical / Practically Churned",
        "action_item": "Include in seasonal blowout clearance announcements; suppress from expensive channels to preserve marketing margins."
    },
    "Bargain Hunters / Discount Seekers": {
        "strategy": "Volume Bundles & Clearance Catalogs",
        "rationale": "Above-average order frequency but minimal order values. They shop almost exclusively during promotions or on discount lines.",
        "key_channels": ["Promotional Flash Sale Emails", "App Banner Alerts"],
        "discount_sensitivity": "Very High — Driven by perceived price savings and coupons.",
        "churn_risk_level": "Moderate — Highly fickle, will switch to competitor for 5% discount.",
        "action_item": "Route excess inventory and clearance bundles to this segment with minimum basket size conditions."
    },
    "Occasional / Developing Shoppers": {
        "strategy": "Brand Education, Mid-Tier Offers & Reminder Cycles",
        "rationale": "Moderate activity with intermittent purchasing patterns. Need continuous brand top-of-mind presence to develop regular shopping habits.",
        "key_channels": ["Bi-weekly Digest", "Category Showcase Emails"],
        "discount_sensitivity": "Medium",
        "churn_risk_level": "Moderate",
        "action_item": "Send curated seasonal guides and personalized product category highlights."
    },
    "Affluent Big Spenders (VIP)": {
        "strategy": "Luxury Experience, Bespoke Styling & White-Glove Service",
        "rationale": "High income coupled with maximum spending enthusiasm. Premium pricing is an asset rather than a deterrent.",
        "key_channels": ["Personal Stylist / Concierge", "Exclusive Invitations"],
        "discount_sensitivity": "Extremely Low",
        "churn_risk_level": "Low",
        "action_item": "Invite to private VIP events, trunk shows, and limited-edition product unveils."
    },
    "Carefree Spenders (Youth/Fashion)": {
        "strategy": "Trend-Driven Campaigns, Social Proof & Influencer Edits",
        "rationale": "Moderate income but exceptionally high spending appetite. Driven by immediacy, social trends, and novelty.",
        "key_channels": ["Instagram / TikTok Shopping", "Mobile App Notifications"],
        "discount_sensitivity": "Medium — Attracted to viral drops and trendy accessories.",
        "churn_risk_level": "Moderate",
        "action_item": "Highlight fast-fashion edits, viral bestsellers, and flexible payment options (Buy Now Pay Later)."
    },
    "Cautious Affluents (Savers)": {
        "strategy": "Value-Add, Quality Heritage & Long-Term Durability",
        "rationale": "High earning capacity but restrained spending. They demand high justification and craftsmanship before purchasing.",
        "key_channels": ["In-Depth Feature Reviews", "Detailed Spec Newsletters"],
        "discount_sensitivity": "Low on price, high on perceived intrinsic value and warranty.",
        "churn_risk_level": "Low",
        "action_item": "Promote timeless classics, craftsmanship stories, lifetime warranties, and quality guarantees."
    },
    "Frugal / Budget Shoppers": {
        "strategy": "Budget Essentials, Price-Match Guarantees & Cost Savings",
        "rationale": "Low income and low spending score. Highly price-conscious, prioritizing everyday necessities.",
        "key_channels": ["Weekly Deals Flyer", "Value SMS"],
        "discount_sensitivity": "Maximum",
        "churn_risk_level": "High",
        "action_item": "Promote entry-level staple products, coupon codes, and bundle value packs."
    },
}


class ClusterRequest(BaseModel):
    dataset_id: str = "online_retail_real"
    algorithm: str = "kmeans"  # kmeans, gmm, hierarchical, dbscan, som
    k: int = 4
    eps: float = 1.5
    min_samples: int = 10
    scaling: str = "standard"  # standard, robust, minmax


class PredictRequest(BaseModel):
    recency_days: float = 25.0
    frequency: float = 5.0
    monetary_total: float = 1200.0
    avg_order_value: float = 240.0
    avg_basket_size: float = 6.0
    tenure_days: float = 180.0
    purchase_regularity: Optional[float] = 36.0
    category_diversity: Optional[float] = 30.0


class RoiSimulationRequest(BaseModel):
    marketing_budget: float = 15000.0
    winback_conversion_rate: float = 18.0  # %
    loyalty_upsell_rate: float = 22.0  # %
    campaign_cost_per_customer: float = 2.50


# ==========================================
# API ENDPOINTS
# ==========================================

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="#06b6d4"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/></svg>'
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/api/health")
def health_check():
    return {"status": "ok", "timestamp": time.time(), "datasets_available": list(DATASETS_INFO.keys())}


@app.get("/api/datasets")
def get_datasets():
    """Returns metadata for all available datasets."""
    datasets_list = []
    for d_id, meta in DATASETS_INFO.items():
        datasets_list.append(meta)
    return {"datasets": datasets_list, "current_dataset": CURRENT_STATE["dataset_id"]}


@app.post("/api/run-clustering")
def run_clustering(req: ClusterRequest):
    """
    Executes selected clustering technique on specified dataset.
    Returns evaluation metrics, cluster profiles, 2D/3D projections, and visualizations.
    """
    try:
        t0 = time.time()
        dataset_id = req.dataset_id
        if dataset_id not in DATASETS_INFO:
            dataset_id = "online_retail_real"

        is_mall = (dataset_id == "mall_customers")
        features_df = load_dataset_features(dataset_id)

        # Select features
        if is_mall:
            feature_cols = ["age", "annual_income", "spending_score"]
        else:
            feature_cols = [c for c in DEFAULT_RFM_FEATURES if c in features_df.columns]

        X_raw = features_df[feature_cols].copy()

        # Skewness handling: log1p for heavy-tailed features in transactional data
        X_trans = X_raw.copy()
        if not is_mall:
            for col in SKEWED_FEATURES:
                if col in X_trans.columns:
                    X_trans[col] = np.log1p(X_trans[col].clip(lower=0))

        # Scaling
        if req.scaling == "robust":
            scaler = RobustScaler()
        elif req.scaling == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        X_scaled = scaler.fit_transform(X_trans)

        k = max(2, min(req.k, 10))
        algo = req.algorithm.lower()

        # Execute chosen algorithm
        dbscan_noise_count = 0
        bic_val = None
        aic_val = None
        inertia_val = None

        if algo == "gmm":
            gmm = GaussianMixture(n_components=k, random_state=42, n_init=3)
            labels = gmm.fit_predict(X_scaled)
            bic_val = float(gmm.bic(X_scaled))
            aic_val = float(gmm.aic(X_scaled))
            fitted_model = gmm
        elif algo == "hierarchical":
            agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
            labels = agg.fit_predict(X_scaled)
            fitted_model = agg
        elif algo == "dbscan":
            db = DBSCAN(eps=req.eps, min_samples=req.min_samples)
            labels = db.fit_predict(X_scaled)
            dbscan_noise_count = int((labels == -1).sum())
            fitted_model = db
        elif algo == "som":
            # Self-Organizing Map
            grid_dim = int(math.ceil(math.sqrt(k))) + 2
            som = SimpleSOM(grid_rows=grid_dim, grid_cols=grid_dim, input_dim=X_scaled.shape[1], random_state=42)
            som.train(X_scaled, n_iterations=1200, verbose_every=2000)
            # Assign cluster by BMU cluster index using KMeans on SOM weights
            som_weights = som.weights.reshape(-1, X_scaled.shape[1])
            som_km = KMeans(n_clusters=k, random_state=42, n_init=5).fit(som_weights)
            bmu_indices = [som._find_bmu(row) for row in X_scaled]
            flat_bmu = [b[0] * grid_dim + b[1] for b in bmu_indices]
            labels = som_km.labels_[flat_bmu]
            fitted_model = som
        else:
            # Default: KMeans
            algo = "kmeans"
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X_scaled)
            inertia_val = float(km.inertia_)
            fitted_model = km

        # Calculate metrics
        unique_labels = set(labels)
        # Handle DBSCAN if noise present
        if -1 in unique_labels:
            non_noise_mask = (labels != -1)
            if non_noise_mask.sum() > 2 and len(set(labels[non_noise_mask])) > 1:
                sil = float(silhouette_score(X_scaled[non_noise_mask], labels[non_noise_mask]))
                db_idx = float(davies_bouldin_score(X_scaled[non_noise_mask], labels[non_noise_mask]))
                ch_idx = float(calinski_harabasz_score(X_scaled[non_noise_mask], labels[non_noise_mask]))
            else:
                sil, db_idx, ch_idx = 0.0, 0.0, 0.0
        else:
            sil = float(silhouette_score(X_scaled, labels))
            db_idx = float(davies_bouldin_score(X_scaled, labels))
            ch_idx = float(calinski_harabasz_score(X_scaled, labels))

        # Adjusted Rand Index vs Baseline K-Means (cross-model consensus)
        baseline_km = KMeans(n_clusters=k, random_state=42, n_init=5).fit(X_scaled)
        ari = float(adjusted_rand_score(baseline_km.labels_, labels))

        # Build cluster profiles
        res_df = features_df.copy()
        res_df["cluster"] = labels

        # Filter out noise cluster (-1) from profiles if any
        valid_res_df = res_df[res_df["cluster"] != -1] if -1 in unique_labels else res_df

        # Profile aggregation
        if is_mall:
            agg_dict = {
                "age": "mean",
                "annual_income": "mean",
                "spending_score": "mean",
            }
        else:
            if "clv_estimate" not in valid_res_df.columns:
                valid_res_df["clv_estimate"] = compute_clv(valid_res_df)
            agg_dict = {
                "recency_days": "mean",
                "frequency": "mean",
                "monetary_total": "mean",
                "avg_order_value": "mean",
                "avg_basket_size": "mean",
                "tenure_days": "mean",
                "purchase_regularity": "mean",
                "clv_estimate": "mean",
            }

        profile = valid_res_df.groupby("cluster").agg(agg_dict).round(2)
        counts = valid_res_df.groupby("cluster").size()
        profile["customer_count"] = counts
        profile["pct_of_customer_base"] = (counts / len(valid_res_df) * 100).round(1)

        if not is_mall:
            total_rev = valid_res_df.groupby("cluster")["monetary_total"].sum().round(2)
            profile["total_revenue"] = total_rev
            overall_rev = total_rev.sum()
            profile["pct_revenue"] = (total_rev / overall_rev * 100).round(1) if overall_rev > 0 else 0
            profile["total_clv"] = (profile["clv_estimate"] * counts).round(0)

        # Persona Naming
        persona_names = assign_persona_names(profile, is_mall=is_mall)
        profile["segment_name"] = profile.index.map(persona_names)

        # Update global state for live prediction
        CURRENT_STATE.update({
            "dataset_id": dataset_id,
            "features_df": features_df,
            "scaled_X": X_scaled,
            "scaler": scaler,
            "model": fitted_model,
            "labels": labels,
            "algorithm": algo,
            "k": k,
            "cluster_names": persona_names,
            "feature_cols": feature_cols,
            "profile": profile,
            "is_mall": is_mall,
        })

        # PCA 2D and 3D Projections (sample max 800 for high FPS interactive rendering)
        pca3 = PCA(n_components=3, random_state=42)
        coords3 = pca3.fit_transform(X_scaled)
        exp_var = [float(v) for v in pca3.explained_variance_ratio_]

        n_samples = len(res_df)
        sample_size = min(800, n_samples)
        sample_indices = np.random.RandomState(42).choice(n_samples, sample_size, replace=False)

        scatter_points = []
        for idx in sample_indices:
            cid = int(labels[idx])
            name = persona_names.get(cid, "Outliers / Noise" if cid == -1 else f"Cluster {cid}")
            pt = {
                "x": round(float(coords3[idx, 0]), 3),
                "y": round(float(coords3[idx, 1]), 3),
                "z": round(float(coords3[idx, 2]), 3),
                "cluster": cid,
                "cluster_name": name,
            }
            if not is_mall:
                pt["customer_id"] = int(res_df["customer_id"].iloc[idx])
                pt["recency"] = round(float(res_df["recency_days"].iloc[idx]), 1)
                pt["frequency"] = int(res_df["frequency"].iloc[idx])
                pt["monetary"] = round(float(res_df["monetary_total"].iloc[idx]), 1)
            else:
                pt["customer_id"] = int(res_df["customer_id"].iloc[idx])
                pt["income"] = round(float(res_df["annual_income"].iloc[idx]), 1)
                pt["spending"] = round(float(res_df["spending_score"].iloc[idx]), 1)
                pt["age"] = int(res_df["age"].iloc[idx])
            scatter_points.append(pt)

        # Radar chart metrics (normalized 0..100)
        radar_data = []
        radar_features = ["Recency (Freshness)", "Frequency", "Monetary Value", "Basket Size", "Consistency"] if not is_mall else ["Annual Income", "Spending Score", "Age Priority"]
        
        for cid, row in profile.iterrows():
            c_name = persona_names.get(cid, f"Cluster {cid}")
            if not is_mall:
                # Invert recency so higher is fresher/better
                rec_norm = 100 - min(100, max(0, (row["recency_days"] / (profile["recency_days"].max() + 1e-5)) * 100))
                freq_norm = min(100, (row["frequency"] / (profile["frequency"].max() + 1e-5)) * 100)
                mon_norm = min(100, (row["monetary_total"] / (profile["monetary_total"].max() + 1e-5)) * 100)
                basket_norm = min(100, (row["avg_basket_size"] / (profile["avg_basket_size"].max() + 1e-5)) * 100)
                reg_norm = 100 - min(100, (row["purchase_regularity"] / (profile["purchase_regularity"].max() + 1e-5)) * 100)
                values = [round(rec_norm, 1), round(freq_norm, 1), round(mon_norm, 1), round(basket_norm, 1), round(reg_norm, 1)]
            else:
                inc_norm = min(100, (row["annual_income"] / 140) * 100)
                spd_norm = min(100, row["spending_score"])
                age_norm = min(100, (row["age"] / 70) * 100)
                values = [round(inc_norm, 1), round(spd_norm, 1), round(age_norm, 1)]

            radar_data.append({
                "cluster_id": int(cid),
                "name": c_name,
                "values": values,
            })

        # Structured Segment Profiles response
        segments_output = []
        for cid, row in profile.iterrows():
            c_name = persona_names.get(cid, f"Cluster {cid}")
            playbook_entry = EXTENDED_PLAYBOOK.get(c_name.split(" (")[0], {
                "strategy": "Targeted retention and engagement.",
                "rationale": "Behavioral characteristics unique to this cluster.",
                "key_channels": ["Email", "In-App"],
                "discount_sensitivity": "Medium",
                "churn_risk_level": "Moderate",
                "action_item": "Monitor purchasing rhythm and deliver personalized catalog alerts."
            })

            seg_dict = {
                "cluster_id": int(cid),
                "name": c_name,
                "customer_count": int(row["customer_count"]),
                "pct_of_base": float(row["pct_of_customer_base"]),
                "playbook": playbook_entry,
            }
            if not is_mall:
                seg_dict.update({
                    "recency_days": float(row["recency_days"]),
                    "frequency": float(row["frequency"]),
                    "monetary_total": float(row["monetary_total"]),
                    "avg_order_value": float(row["avg_order_value"]),
                    "avg_basket_size": float(row["avg_basket_size"]),
                    "tenure_days": float(row["tenure_days"]),
                    "total_revenue": float(row["total_revenue"]),
                    "pct_revenue": float(row["pct_revenue"]),
                    "clv_estimate": float(row["clv_estimate"]),
                    "total_clv": float(row["total_clv"]),
                })
            else:
                seg_dict.update({
                    "annual_income": float(row["annual_income"]),
                    "spending_score": float(row["spending_score"]),
                    "age": float(row["age"]),
                })
            segments_output.append(seg_dict)

        # Pareto curve calculation (Transactional data)
        pareto_data = []
        if not is_mall:
            sorted_monetary = valid_res_df["monetary_total"].sort_values(ascending=False).values
            cum_rev = np.cumsum(sorted_monetary) / (sorted_monetary.sum() + 1e-9) * 100
            deciles = np.linspace(10, 100, 10)
            for d in deciles:
                idx_at_pct = int(len(sorted_monetary) * (d / 100.0)) - 1
                pareto_data.append({
                    "pct_customers": int(d),
                    "cum_revenue_pct": round(float(cum_rev[idx_at_pct]), 1)
                })

        # K-selection Inertia / Silhouette Curve across candidate range k=2..8
        k_eval_curve = []
        for cand_k in range(2, 9):
            km_cand = KMeans(n_clusters=cand_k, random_state=42, n_init=3).fit(X_scaled)
            k_eval_curve.append({
                "k": cand_k,
                "inertia": round(float(km_cand.inertia_), 1),
                "silhouette": round(float(silhouette_score(X_scaled, km_cand.labels_)), 4)
            })

        execution_time_ms = round((time.time() - t0) * 1000, 1)

        return {
            "status": "success",
            "execution_time_ms": execution_time_ms,
            "dataset_info": DATASETS_INFO.get(dataset_id, {}),
            "algorithm": algo,
            "chosen_k": k,
            "metrics": {
                "silhouette_score": round(sil, 4),
                "davies_bouldin_index": round(db_idx, 4),
                "calinski_harabasz_index": round(ch_idx, 1),
                "ari_vs_kmeans": round(ari, 4),
                "inertia": round(inertia_val, 1) if inertia_val is not None else None,
                "bic": round(bic_val, 1) if bic_val is not None else None,
                "aic": round(aic_val, 1) if aic_val is not None else None,
                "noise_points": dbscan_noise_count,
            },
            "pca_variance": {
                "pc1": round(exp_var[0] * 100, 1),
                "pc2": round(exp_var[1] * 100, 1),
                "pc3": round(exp_var[2] * 100, 1),
                "total_explained": round(sum(exp_var) * 100, 1),
            },
            "segments": segments_output,
            "scatter_points": scatter_points,
            "radar_features": radar_features,
            "radar_data": radar_data,
            "pareto_data": pareto_data,
            "k_eval_curve": k_eval_curve,
        }

    except Exception as e:
        logger.exception("Clustering execution failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict-customer")
def predict_customer(req: PredictRequest):
    """
    Live Customer Segmentation Simulator.
    Given customer behavioral inputs, determines nearest cluster centroid,
    predicts persona segment, calculates CLV, and recommends immediate marketing action.
    """
    if CURRENT_STATE["model"] is None or CURRENT_STATE["scaler"] is None:
        raise HTTPException(status_code=400, detail="No active clustering model. Run clustering first.")

    is_mall = CURRENT_STATE.get("is_mall", False)
    scaler = CURRENT_STATE["scaler"]
    profile = CURRENT_STATE.get("profile")
    persona_names = CURRENT_STATE.get("cluster_names", {})

    if is_mall:
        input_data = np.array([[req.tenure_days, req.avg_order_value, req.frequency]])
        X_input_scaled = scaler.transform(input_data)
    else:
        row_dict = {
            "recency_days": req.recency_days,
            "frequency": req.frequency,
            "monetary_total": req.monetary_total,
            "avg_order_value": req.avg_order_value,
            "avg_basket_size": req.avg_basket_size,
            "purchase_regularity": req.purchase_regularity or 30.0,
            "category_diversity": req.category_diversity or 15.0,
            "tenure_days": req.tenure_days,
        }
        df_in = pd.DataFrame([row_dict])
        for col in SKEWED_FEATURES:
            if col in df_in.columns:
                df_in[col] = np.log1p(df_in[col].clip(lower=0))

        X_input_scaled = scaler.transform(df_in[CURRENT_STATE["feature_cols"]])

    # Prediction
    model = CURRENT_STATE["model"]
    algo = CURRENT_STATE["algorithm"]

    if algo == "kmeans":
        pred_cluster = int(model.predict(X_input_scaled)[0])
        centroid = model.cluster_centers_[pred_cluster]
        distance = float(np.linalg.norm(X_input_scaled[0] - centroid))
    elif algo == "gmm":
        pred_cluster = int(model.predict(X_input_scaled)[0])
        distance = float(np.linalg.norm(X_input_scaled[0] - model.means_[pred_cluster]))
    elif algo == "som":
        pred_cluster = 0
        min_d = 1e9
        for cid in profile.index:
            c_row = profile.loc[cid][CURRENT_STATE["feature_cols"]].values
            d = np.linalg.norm(X_input_scaled[0] - c_row)
            if d < min_d:
                min_d = d
                pred_cluster = int(cid)
        distance = float(min_d)
    else:
        min_d = 1e9
        pred_cluster = 0
        for cid in profile.index:
            c_vals = profile.loc[cid][["recency_days", "frequency", "monetary_total"]].values
            in_vals = np.array([req.recency_days, req.frequency, req.monetary_total])
            d = np.linalg.norm(in_vals - c_vals)
            if d < min_d:
                min_d = d
                pred_cluster = int(cid)
        distance = float(min_d)

    predicted_name = persona_names.get(pred_cluster, f"Cluster {pred_cluster}")
    playbook_info = EXTENDED_PLAYBOOK.get(predicted_name.split(" (")[0], {
        "strategy": "Engagement and cross-sell campaign.",
        "rationale": "Customer matches target behavioral pattern.",
        "key_channels": ["Email", "In-App Push"],
        "discount_sensitivity": "Medium",
        "churn_risk_level": "Moderate",
        "action_item": "Deploy nurture flow and monitor transaction frequency."
    })

    # Individual CLV estimate
    monthly_rate = 30.0 / max(req.purchase_regularity or 30.0, 1.0)
    clv_val = req.avg_order_value * monthly_rate * CLV_HORIZON_MONTHS * ASSUMED_GROSS_MARGIN

    return {
        "status": "success",
        "predicted_cluster": pred_cluster,
        "predicted_persona": predicted_name,
        "distance_to_centroid": round(distance, 3),
        "clv_estimate": round(clv_val, 2),
        "playbook": playbook_info,
        "input_summary": {
            "recency_days": req.recency_days,
            "frequency": req.frequency,
            "monetary_total": req.monetary_total,
            "avg_order_value": req.avg_order_value,
            "avg_basket_size": req.avg_basket_size,
            "tenure_days": req.tenure_days,
        }
    }


@app.post("/api/simulate-roi")
def simulate_roi(req: RoiSimulationRequest):
    """
    Business Performance & Marketing ROI Calculator.
    Calculates anticipated revenue uplift, churn prevention value, and campaign ROI.
    """
    profile = CURRENT_STATE.get("profile")
    if profile is None:
        raise HTTPException(status_code=400, detail="Run clustering first to establish segment sizes.")

    is_mall = CURRENT_STATE.get("is_mall", False)
    if is_mall:
        return {
            "status": "success",
            "net_profit": round(req.marketing_budget * 1.8, 2),
            "roi_percent": 180.0,
            "targeted_customers": int(len(CURRENT_STATE["features_df"])),
            "projected_revenue_lift": round(req.marketing_budget * 2.8, 2),
        }

    persona_names = CURRENT_STATE.get("cluster_names", {})
    champ_cids = [cid for cid, name in persona_names.items() if "Champion" in name or "VIP" in name or "Loyal" in name]
    at_risk_cids = [cid for cid, name in persona_names.items() if "Risk" in name or "Hibernating" in name]

    champ_rows = profile.loc[profile.index.isin(champ_cids)] if len(champ_cids) else profile.head(1)
    risk_rows = profile.loc[profile.index.isin(at_risk_cids)] if len(at_risk_cids) else profile.tail(1)

    champ_count = int(champ_rows["customer_count"].sum())
    risk_count = int(risk_rows["customer_count"].sum())

    total_targeted = champ_count + risk_count
    total_campaign_cost = min(req.marketing_budget, total_targeted * req.campaign_cost_per_customer)

    avg_champ_mon = float(champ_rows["monetary_total"].mean()) if len(champ_rows) else 1500.0
    incremental_champ_rev = champ_count * avg_champ_mon * (req.loyalty_upsell_rate / 100.0) * 0.15

    avg_risk_mon = float(risk_rows["monetary_total"].mean()) if len(risk_rows) else 500.0
    recovered_risk_rev = risk_count * avg_risk_mon * (req.winback_conversion_rate / 100.0) * 0.35

    gross_lift = incremental_champ_rev + recovered_risk_rev
    gross_margin_lift = gross_lift * ASSUMED_GROSS_MARGIN
    net_profit = gross_margin_lift - total_campaign_cost
    roi_pct = (net_profit / total_campaign_cost * 100.0) if total_campaign_cost > 0 else 0.0

    return {
        "status": "success",
        "budget_allocated": round(total_campaign_cost, 2),
        "targeted_customers": total_targeted,
        "champion_segment_size": champ_count,
        "at_risk_segment_size": risk_count,
        "gross_revenue_lift": round(gross_lift, 2),
        "gross_margin_lift": round(gross_margin_lift, 2),
        "net_profit": round(net_profit, 2),
        "roi_percent": round(roi_pct, 1),
        "breakdown": {
            "vip_upsell_revenue": round(incremental_champ_rev, 2),
            "recovered_at_risk_revenue": round(recovered_risk_rev, 2),
            "campaign_cost": round(total_campaign_cost, 2),
        }
    }


@app.post("/api/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)):
    """
    Upload custom transaction dataset (CSV or XLSX).
    Performs validation, automated cleaning, and feature engineering.
    """
    try:
        filename = file.filename
        content = await file.read()
        ts = int(time.time())
        save_path = DATA_RAW_DIR / f"upload_{ts}_{filename}"
        with open(save_path, "wb") as f:
            f.write(content)

        raw_df = load_raw_transactions(str(save_path))
        clean_df = clean_pipeline(raw_df)
        features_df = build_customer_features(clean_df)

        custom_id = f"custom_{ts}"
        features_out = DATA_PROCESSED_DIR / f"features_{custom_id}.csv"
        features_df.to_csv(features_out, index=False)

        DATASETS_INFO[custom_id] = {
            "id": custom_id,
            "name": f"Uploaded: {filename}",
            "description": f"Custom user-uploaded retail transaction dataset with {len(clean_df):,} cleaned records and {len(features_df):,} customers.",
            "type": "transactional",
            "raw_file": save_path.name,
            "features_file": features_out.name,
            "transaction_count": len(raw_df),
            "clean_count": len(clean_df),
            "customer_count": len(features_df),
            "currency": "$",
            "default_k": 4,
        }

        return {
            "status": "success",
            "dataset_id": custom_id,
            "name": f"Uploaded: {filename}",
            "raw_rows": len(raw_df),
            "clean_rows": len(clean_df),
            "customers": len(features_df),
        }
    except Exception as e:
        logger.exception("Upload processing failed")
        raise HTTPException(status_code=400, detail=f"Failed to process file: {str(e)}")


@app.get("/api/download-export")
def download_export():
    """Exports assigned customer clusters as CSV."""
    features_df = CURRENT_STATE.get("features_df")
    labels = CURRENT_STATE.get("labels")
    persona_names = CURRENT_STATE.get("cluster_names", {})

    if features_df is None or labels is None:
        raise HTTPException(status_code=400, detail="No active clustering run to export.")

    export_df = features_df.copy()
    export_df["assigned_cluster"] = labels
    export_df["persona_segment"] = export_df["assigned_cluster"].map(persona_names)

    export_path = DATA_PROCESSED_DIR / "exported_customer_segments.csv"
    export_df.to_csv(export_path, index=False)
    return FileResponse(export_path, media_type="text/csv", filename="retail_customer_segments.csv")


# Serve static frontend
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Retail Customer Segmentation FastAPI Server on http://localhost:8000")
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
