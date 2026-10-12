# Retail Customer Segmentation

Behavioral segmentation of retail customers using transaction-level purchase
data, built for personalized marketing and business decision-making.

> **Course context:** ML-2, CO3 (unsupervised learning / clustering). K-Means
> is the primary technique (syllabus Module 3), cross-validated against a
> Gaussian Mixture Model, with PCA used for visualization.

---

## 1. Project goals

- Segment customers by *purchasing behavior*, not demographics (which the
  dataset doesn't have) — this is the industry-standard approach when
  demographic data is unavailable or unreliable.
- Make every cleaning, feature, and modeling decision explicit and
  justified, not just "run K-Means and report clusters."
- Translate clusters into segments a marketing team could actually act on:
  named segments, quantified value (CLV), and a specific playbook per segment.

---

## 2. Dataset

**[Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)**
(UCI Machine Learning Repository) — ~1 million line-item transactions from a
UK-based online retailer, Dec 2009–Dec 2011. Chosen over alternatives
(Mall Customers, Instacart, pre-engineered Kaggle sets) because it's **raw
transactional data**, which forces genuine data cleaning and feature
engineering rather than starting from a dataset someone already summarized —
that's the part of this project that actually resembles real analytics work.

### Getting the real data (for your final run)

1. Download `online_retail_II.xlsx` from the UCI link above.
2. It ships as two sheets ("Year 2009-2010", "Year 2010-2011"). Combine and
   export to CSV with this one-time script:

```python
import pandas as pd
df1 = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2009-2010")
df2 = pd.read_excel("online_retail_II.xlsx", sheet_name="Year 2010-2011")
pd.concat([df1, df2]).to_csv("data/raw/online_retail_ii.csv", index=False)
```

3. Confirm the column names match exactly: `Invoice, StockCode, Description,
   Quantity, InvoiceDate, Price, Customer ID, Country`. This is the UCI
   schema as of the current release — if it doesn't match, adjust the
   `COL_*` constants in `src/config.py` rather than renaming columns
   throughout the codebase.

### Why this repo currently runs on synthetic data

This environment has no internet access, so the real 40MB file couldn't be
downloaded during development. `src/generate_sample_data.py` generates a
**schema-identical synthetic sample** (1,500 customers, ~46K transactions)
with the same messiness patterns as the real data — cancellations, missing
Customer IDs, free-item price errors — so the full pipeline could be built
and validated end-to-end. **Every number in this README's results section is
from that synthetic run.** Swap in the real file (same filename, same path)
and rerun `python3 src/main.py` — nothing else changes. Do this before final
submission; the real dataset will give you your actual results and is what
you should present.

---

## 3. Repository structure

```
retail-segmentation/
├── data/
│   ├── raw/                    # input transaction file
│   └── processed/              # cleaned data, features, cluster assignments
├── src/
│   ├── config.py                # every constant/threshold, single source of truth
│   ├── data_loader.py            # schema-validated ingestion
│   ├── preprocessing.py          # cleaning, with logged before/after row counts
│   ├── feature_engineering.py    # RFM + behavioral features
│   ├── clustering.py             # k selection, K-Means, GMM validation, PCA
│   ├── profiling.py              # segment naming, CLV, marketing playbook
│   ├── visualization.py          # all report figures
│   ├── generate_sample_data.py   # synthetic data generator (dev/testing only)
│   └── main.py                   # orchestrates the full pipeline
├── outputs/
│   ├── figures/                  # 6 report-ready PNG figures
│   └── reports/
├── dashboard/
│   ├── index.html                 # interactive segment dashboard
│   └── segment_summary.json       # data consumed by the dashboard
└── README.md
```

Each `src/` module can also be run standalone (`python3 feature_engineering.py`)
for debugging a single stage without rerunning the whole pipeline.

## 4. How to run

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

# (only needed if you haven't already got the real dataset in place)
python3 src/generate_sample_data.py

python3 src/main.py
```

This produces: cleaned transactions, customer feature table, cluster
assignments, 6 figures in `outputs/figures/`, and `dashboard/segment_summary.json`.
Open `dashboard/index.html` in a browser to view the interactive segment
explorer (no server needed — it's self-contained).

---

## 5. Methodology

### 5.1 Data cleaning

Raw retail transaction exports are messy in specific, well-known ways. Each
is handled explicitly rather than with a blanket `dropna()`:

| Issue | Handling | Why |
|---|---|---|
| Cancelled orders (`Invoice` starts with `C`) | Removed | Segmentation should reflect realized purchases, not net-of-cancellation totals |
| Missing `Customer ID` | Removed | Can't attribute a transaction to a customer identity — imputing an ID is meaningless |
| Non-positive quantity/price | Removed | Data entry errors or free promotional items that would distort monetary features |
| Admin line items (postage, bank charges, etc.) | Removed | Not genuine product purchases |
| Extreme quantity/price outliers | **Capped** at 99.5th percentile, not deleted | Usually genuine bulk buyers, not errors — but left uncapped they destabilize K-Means centroids (Euclidean distance is outlier-sensitive) |

Every step logs exactly how many rows it removed and why (see
`preprocessing.py` console output) — this is what makes the cleaning process
auditable rather than a black box.

### 5.2 Feature engineering

One row per customer, built from:

- **RFM** — Recency (days since last order), Frequency (distinct orders),
  Monetary (total spend). The industry-standard baseline for retail
  segmentation; every commercial CRM (Salesforce, HubSpot, Klaviyo) computes
  some version of this.
- **Order shape** — average order value, average basket size (distinct
  items per order).
- **Behavioral rhythm** — purchase regularity (average days between orders),
  tenure (days between first and last purchase).
- **Assortment** — category diversity (distinct products ever purchased).

**Preprocessing before clustering:** RFM-style features are heavily
right-skewed (a few customers spend far more than the median). K-Means uses
Euclidean distance, so `monetary_total` (range: tens to thousands) would
otherwise dominate a bounded feature like `category_diversity`. Skewed
features are log-transformed, then everything is standard-scaled.

**A subtlety worth noting explicitly:** `purchase_regularity` for one-time
buyers has no observed value (there's no gap between orders to measure). An
early version of this pipeline defaulted it to their own (near-zero) tenure,
which wrongly implied one-time buyers purchase extremely often, and
silently inflated their estimated CLV above genuinely loyal customers. It's
now imputed conservatively with the full observation window — "we have zero
evidence of a repeat cadence" — and is called out here because it's exactly
the kind of subtle feature-engineering bug that produces confidently wrong
segments if untested against actual output.

### 5.3 Clustering

- **k selection:** K-Means fit for k = 2–10; both inertia (elbow) and
  silhouette score are computed for each. The purely statistical optimum
  (max silhouette) is reported, but if it falls outside a business-usable
  range (fewer than 3 segments — you can't run differentiated campaigns off
  2 groups), the best silhouette **within** that range is selected instead.
  This trade-off is logged, not hidden.
- **Validation against a second algorithm:** a Gaussian Mixture Model is fit
  at the same k. Adjusted Rand Index between K-Means and GMM labels measures
  how much two structurally different algorithms agree on the grouping — a
  single algorithm's output with no cross-check is not sufficient evidence
  the structure is real rather than an artifact of that algorithm's
  assumptions (K-Means assumes spherical, similarly-sized clusters; GMM
  doesn't).
- **PCA** projects the scaled feature space to 2D purely for visualization.
  Clustering itself happens in the full feature space — PCA is not used to
  reduce dimensionality before clustering, only to plot the result.

### 5.4 Business translation

- **Segment naming** uses continuous, min-max normalized RFM scores
  (composited into a single 0–1 score), not fixed rank-position thresholds —
  which behave poorly when k is small. A secondary check on average order
  value distinguishes segments that look similar on RFM alone but have very
  different per-order value (e.g. "At-Risk High Value" vs. "Hibernating" —
  same low recency/frequency, but one still spends well per order).
- **CLV proxy:** `avg_order_value × (30 / purchase_regularity) × 12 months ×
  30% assumed margin`. This is a simple linear projection, not a
  probabilistic model (BG/NBD or Pareto/NBD would be the natural extension —
  see Section 7).
- **Marketing playbook:** each named segment archetype maps to a specific
  strategy and a stated rationale (`profiling.py: MARKETING_PLAYBOOK`), so
  the deliverable is a decision, not just a description.

### 5.5 Self-Organizing Map (bonus — syllabus Module 4)

A Self-Organizing Map is trained **from scratch** (`src/som.py` — no
library, to mirror the algorithm as taught: weight initialization,
best-matching-unit selection via competitive learning, Gaussian
neighbourhood updates, learning-rate/radius decay) on the same scaled
feature matrix used for K-Means.

The SOM is not used as the segmentation method itself — K-Means remains
primary. Instead it serves as a **second, structurally independent
cross-check**: if K-Means's centroid-based segments occupy coherent,
separated regions of a map built by an entirely different mechanism
(competitive learning, no notion of centroids or fixed k), that's further
evidence the segments reflect real structure in customer behavior rather
than an artifact of K-Means's own assumptions. On the sample run, the three
K-Means segments land in clearly separated horizontal bands of the SOM grid
(see `outputs/figures/07_som_map.png`) — a second confirmation alongside
the GMM/ARI check in Section 5.3.

---

## 6. Results (synthetic sample — replace with real-data run for submission)

- **1,493 customers**, 41,283 cleaned transactions, Dec 2010–Dec 2011.
- **k = 3** selected (silhouette = 0.367; the statistically-optimal k=2 was
  set aside as insufficiently granular for differentiated marketing).
- **Adjusted Rand Index = 0.85** between K-Means and GMM — strong agreement,
  good evidence the segments reflect real structure.
- **PCA 2D projection explains 75% of feature variance** — the segments are
  visually well-separated, not an artifact of a high-dimensional space that
  looks fine in 2D but is actually noise.

| Segment | Size | Avg. Recency | Avg. Orders | 12mo CLV/customer | Strategy |
|---|---|---|---|---|---|
| Champions | 909 (60.9%) | 60 days | 11.2 | £390 | Loyalty programs, VIP tiers, referral incentives |
| At-Risk High Value | 392 (26.3%) | 162 days | 1.9 | £528 | Personalized win-back campaigns, direct outreach |
| Hibernating / Lost | 192 (12.9%) | 154 days | 1.4 | £96 | Low-cost reactivation only; don't over-invest |

Full figures: `outputs/figures/01_k_selection.png` through `07_som_map.png`.

**Business performance impact:** segmentation lets marketing spend scale
with expected return rather than being uniform. The At-Risk segment shows
why RFM-only analysis (which would rank it below Champions on every raw
metric) misses the point — its *per-customer* CLV is actually the highest
of the three groups, meaning it's the highest-ROI group for win-back
investment, not the lowest. A blanket "target your biggest spenders" rule
would have missed this entirely.

---

## 7. Limitations & possible extensions

- **CLV is a linear proxy**, not a fitted probabilistic model. A BG/NBD
  (Beta-Geometric/NBD) model would estimate purchase probability and
  expected future orders per customer properly — a natural next step
  (`lifetimes` Python package).
- **No product-level analysis.** Market basket analysis (association rules)
  on top of this segmentation would answer "what should we cross-sell to
  each segment," not just "who to target."
- **Static segmentation.** In production this would run on a schedule
  (weekly/monthly) with segment-transition tracking (which customers moved
  from Champion → At-Risk), which is often more actionable than a
  point-in-time snapshot.
- **SOM as an alternative view** — done (Section 5.5); a natural further
  step would be a component-plane analysis (one heatmap per feature across
  the SOM grid) to see which specific features drive each map region.
