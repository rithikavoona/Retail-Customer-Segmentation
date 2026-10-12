"""
Translates raw cluster output into business-usable customer segments.

This is the module that separates "we ran K-Means" from "we built a
segmentation system a marketing team could actually use". It does three
things a stakeholder actually cares about:

  1. Names each cluster using RFM-style business heuristics (the same
     vocabulary real CRM platforms like Salesforce/HubSpot use), rather
     than shipping "Cluster 0, Cluster 1, ...".
  2. Estimates a simple Customer Lifetime Value (CLV) per segment, so the
     business impact of each segment can be quantified in currency, not
     just cluster statistics.
  3. Maps each segment to a concrete marketing action.
"""

import logging
import pandas as pd

from config import CLV_HORIZON_MONTHS, ASSUMED_GROSS_MARGIN

logger = logging.getLogger(__name__)


def compute_clv(df: pd.DataFrame) -> pd.Series:
    """
    Simple, transparent CLV proxy (not a full BG/NBD probabilistic model --
    that's a legitimate 'future work' extension, noted in the README):

        CLV = avg_order_value * monthly_purchase_rate * horizon_months * margin

    monthly_purchase_rate is derived from purchase_regularity (days between
    orders) rather than raw frequency, because frequency alone doesn't
    account for how long the customer has actually been active.
    """
    monthly_purchase_rate = 30.0 / df["purchase_regularity"].clip(lower=1)
    clv = df["avg_order_value"] * monthly_purchase_rate * CLV_HORIZON_MONTHS * ASSUMED_GROSS_MARGIN
    return clv


def build_segment_profiles(cluster_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates mean feature values and CLV per cluster -- the raw material
    for naming and interpretation."""
    cluster_df = cluster_df.copy()
    cluster_df["clv_estimate"] = compute_clv(cluster_df)

    agg_cols = ["recency_days", "frequency", "monetary_total", "avg_order_value",
                "avg_basket_size", "purchase_regularity", "category_diversity",
                "tenure_days", "clv_estimate"]

    profile = cluster_df.groupby("cluster")[agg_cols].mean().round(2)
    profile["customer_count"] = cluster_df.groupby("cluster").size()
    profile["pct_of_customer_base"] = (profile["customer_count"] / len(cluster_df) * 100).round(1)
    profile["total_clv"] = (profile["clv_estimate"] * profile["customer_count"]).round(0)
    return profile.sort_values("clv_estimate", ascending=False)


def _minmax(series: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]. Returns all-0.5 if the series is constant
    (avoids divide-by-zero when every cluster happens to tie on a feature)."""
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-9:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / (hi - lo)


def name_segments(profile: pd.DataFrame) -> dict:
    """
    Names each cluster from CONTINUOUS, min-max normalized RFM scores rather
    than rank-position thresholds. Rank-based thresholds (e.g. "top 40% by
    rank") behave poorly when k is small -- with only 3 clusters, rank
    quantiles collapse into coarse buckets and different clusters end up
    with identical labels despite very different underlying values. A
    normalized composite score scales correctly for any k.

    R/F/M are each normalized to [0, 1] across clusters (recency inverted so
    higher = more recent = better), then averaged into a composite score.
    A secondary check on avg_order_value vs. the population median separates
    "Bargain Hunters" (frequent, low order value) from generically loyal
    customers, since that distinction matters for the marketing playbook
    even when it doesn't show up in the RFM composite.
    """
    scored = profile.copy()
    scored["r_score"] = 1 - _minmax(scored["recency_days"])   # invert: recent = high score
    scored["f_score"] = _minmax(scored["frequency"])
    scored["m_score"] = _minmax(scored["monetary_total"])
    scored["rfm_score"] = (scored["r_score"] + scored["f_score"] + scored["m_score"]) / 3

    overall_median_aov = profile["avg_order_value"].median()

    names = {}
    for cluster_id, row in scored.iterrows():
        score = row["rfm_score"]
        is_bargain = (row["avg_order_value"] < overall_median_aov) and (row["f_score"] > 0.5)
        is_new = (row["r_score"] > 0.65) and (row["f_score"] < 0.35)

        if is_new:
            name = "Promising / New"
        elif score >= 0.66:
            name = "Champions"
        elif is_bargain and score >= 0.35:
            name = "Bargain Hunters"
        elif score >= 0.5:
            name = "Loyal / Steady Regulars"
        elif row["avg_order_value"] >= overall_median_aov:
            # Low recency/frequency composite BUT their per-order spend holds
            # up against the population -- this is a customer who used to (or
            # occasionally does) spend well but has drifted, not a low-value
            # customer. That distinction changes the marketing response
            # entirely (win-back investment vs. write-off), so it's checked
            # before falling through to the generic "Hibernating" bucket.
            name = "At-Risk High Value"
        else:
            name = "Hibernating / Lost"

        names[cluster_id] = name

    # Disambiguate only if the SAME name is genuinely reused (rare once
    # scores are continuous, but kept as a safety net for edge cases).
    seen = {}
    final_names = {}
    for cid, name in names.items():
        if name in seen:
            seen[name] += 1
            final_names[cid] = f"{name} ({seen[name]})"
        else:
            seen[name] = 1
            final_names[cid] = name
    return final_names


MARKETING_PLAYBOOK = {
    "Champions": {
        "strategy": "Loyalty & advocacy programs, early access to new products, "
                    "VIP tiers, referral incentives.",
        "rationale": "Highest recency, frequency, and monetary value. Retention "
                     "cost here is far lower than acquisition cost elsewhere, "
                     "and this group has the highest referral potential.",
    },
    "At-Risk High Value": {
        "strategy": "Personalized win-back campaigns, time-limited offers, "
                    "direct outreach (email/call), satisfaction surveys.",
        "rationale": "High historical spend but rising recency (drifting away). "
                     "Losing this group has an outsized revenue impact, so "
                     "proactive intervention has strong expected ROI.",
    },
    "Promising / New": {
        "strategy": "Onboarding email sequences, second-purchase incentives, "
                    "cross-sell recommendations based on first purchase category.",
        "rationale": "Recent but low order count -- the goal is converting a "
                     "first/second purchase into a habit before they churn.",
    },
    "Hibernating / Lost": {
        "strategy": "Low-cost reactivation campaigns only (broad discount emails, "
                    "re-engagement sequences); avoid heavy investment.",
        "rationale": "Long recency and low frequency/monetary value -- the "
                     "economics rarely justify expensive win-back tactics here.",
    },
    "Bargain Hunters": {
        "strategy": "Promotion and clearance alerts, bundle deals, loyalty "
                    "points programs that reward volume over margin.",
        "rationale": "Frequent purchasers but low average order value -- price "
                     "sensitive. Volume-based incentives fit better than premium offers.",
    },
    "Steady Regulars": {
        "strategy": "Standard newsletters, seasonal promotions, moderate "
                    "cross-sell -- maintain engagement without heavy spend.",
        "rationale": "Mid-range across RFM dimensions; stable but with headroom "
                     "to grow into 'Champions' with the right nudge.",
    },
}


def get_playbook(segment_name: str) -> dict:
    """Looks up the marketing playbook, stripping any '(2)' disambiguation suffix."""
    base_name = segment_name.split(" (")[0]
    return MARKETING_PLAYBOOK.get(base_name, {
        "strategy": "Custom analysis recommended -- segment did not map cleanly "
                    "to a standard RFM archetype.",
        "rationale": "Review this segment's raw feature averages manually.",
    })


if __name__ == "__main__":
    from config import CLUSTER_ASSIGNMENTS_FILE

    cluster_df = pd.read_csv(CLUSTER_ASSIGNMENTS_FILE)
    profile = build_segment_profiles(cluster_df)
    names = name_segments(profile)
    profile["segment_name"] = profile.index.map(names)

    pd.set_option("display.width", 150)
    print(profile[["segment_name", "customer_count", "pct_of_customer_base",
                    "recency_days", "frequency", "monetary_total",
                    "clv_estimate", "total_clv"]])

    print("\n--- Marketing Playbook ---")
    for cid, name in names.items():
        pb = get_playbook(name)
        print(f"\n[Cluster {cid}] {name}")
        print(f"  Strategy:  {pb['strategy']}")
        print(f"  Rationale: {pb['rationale']}")
