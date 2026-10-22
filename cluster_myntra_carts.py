"""
cluster_myntra_carts.py

Step 2: Unsupervised clustering of abandoned-cart behavior.

Pipeline:
1. Load myntra_carts.csv (output of Step 1).
2. Select + scale numerical features (StandardScaler).
3. Fit K-Means (n_clusters=5, random_state=42).
4. Automatically map each cluster to a human-readable "archetype" based on
   which feature(s) dominate that cluster's centroid, using the Hungarian
   algorithm for an optimal one-to-one (cluster <-> archetype) assignment.
5. Export:
   - myntra_carts_clustered.csv  (original data + Cluster_ID + Archetype)
   - kmeans_model.pkl            (fitted KMeans model)
   - scaler.pkl                  (fitted StandardScaler)
   - a printed summary table of feature averages / counts per archetype
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_PATH = "myntra_carts.csv"
OUTPUT_PATH = "myntra_carts_clustered.csv"
MODEL_PATH = "kmeans_model.pkl"
SCALER_PATH = "scaler.pkl"

FEATURE_COLS = [
    "Cart_Value",
    "Time_Spent_In_App",
    "Payment_Failures",
    "Previous_Purchases",
    "Discount_Sensitivity",
]

N_CLUSTERS = 5
RANDOM_STATE = 42

ARCHETYPES = [
    "Checkout Friction",
    "Price Sensitive",
    "Window Shopper",
    "High-Intent Loyalist",
    "Casual / Hesitant",
]


# ---------------------------------------------------------------------------
# Step 1: Load data
# ---------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Load the cart dataset from CSV."""
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")
    return df


# ---------------------------------------------------------------------------
# Step 2: Feature selection & preprocessing
# ---------------------------------------------------------------------------
def scale_features(df: pd.DataFrame, feature_cols: list) -> tuple[np.ndarray, StandardScaler]:
    """Standardize numerical features (zero mean, unit variance)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[feature_cols])
    return X_scaled, scaler


# ---------------------------------------------------------------------------
# Step 3: Model training
# ---------------------------------------------------------------------------
def fit_kmeans(X_scaled: np.ndarray, n_clusters: int, random_state: int) -> KMeans:
    """Fit a K-Means model on the scaled feature matrix."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)
    return kmeans


# ---------------------------------------------------------------------------
# Step 4: Automated archetype mapping
# ---------------------------------------------------------------------------
def compute_cluster_profiles(cluster_centers_scaled: np.ndarray, feature_cols: list) -> pd.DataFrame:
    """
    Cluster centroids (already in standardized/z-score space, since a K-Means
    centroid is by definition the mean of its members' scaled features).
    Returns a DataFrame: rows = cluster_id, cols = feature z-scores.
    """
    return pd.DataFrame(cluster_centers_scaled, columns=feature_cols)


def build_archetype_score_matrix(profiles_z: pd.DataFrame) -> pd.DataFrame:
    """
    Build a (cluster x archetype) score matrix. Each archetype's score for a
    given cluster reflects how well that cluster's centroid matches the
    defining behavioral signature of the archetype, using standardized
    (z-score) feature values so scores are directly comparable.

    Higher score = stronger match.
    """
    scores = pd.DataFrame(index=profiles_z.index, columns=ARCHETYPES, dtype=float)

    # Checkout Friction: dominated by high Payment_Failures
    scores["Checkout Friction"] = profiles_z["Payment_Failures"]

    # Price Sensitive: dominated by high Discount_Sensitivity
    scores["Price Sensitive"] = profiles_z["Discount_Sensitivity"]

    # Window Shopper: high time in app, but low cart value (browsing, not buying)
    scores["Window Shopper"] = profiles_z["Time_Spent_In_App"] - profiles_z["Cart_Value"]

    # High-Intent Loyalist: high previous purchases AND high cart value
    scores["High-Intent Loyalist"] = profiles_z["Previous_Purchases"] + profiles_z["Cart_Value"]

    # Casual / Hesitant: low previous purchases and low overall engagement
    # across the board (the "quiet" cluster) -- essentially the inverse of
    # total activity, so it naturally absorbs whichever cluster is flattest.
    total_engagement = profiles_z.sum(axis=1)
    scores["Casual / Hesitant"] = -profiles_z["Previous_Purchases"] - total_engagement

    return scores


def assign_archetypes(score_matrix: pd.DataFrame) -> dict:
    """
    Solve an optimal one-to-one assignment between clusters and archetypes
    using the Hungarian algorithm, maximizing total match quality across
    all 5 clusters simultaneously (rather than greedily picking per-feature
    winners, which could assign the same archetype twice).
    """
    # linear_sum_assignment minimizes cost, so negate scores to maximize them
    cost_matrix = -score_matrix.values
    cluster_idx, archetype_idx = linear_sum_assignment(cost_matrix)

    mapping = {}
    for c_idx, a_idx in zip(cluster_idx, archetype_idx):
        cluster_id = score_matrix.index[c_idx]
        archetype_name = score_matrix.columns[a_idx]
        mapping[cluster_id] = archetype_name
    return mapping


# ---------------------------------------------------------------------------
# Step 5: Reporting
# ---------------------------------------------------------------------------
def print_archetype_summary(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Print + return a summary table: mean feature values and count per archetype."""
    summary = (
        df.groupby("Archetype")[feature_cols]
        .mean()
        .round(2)
        .assign(User_Count=df.groupby("Archetype").size())
        .sort_values("User_Count", ascending=False)
    )
    print("\n=== Archetype Summary (feature averages + user count) ===")
    print(summary.to_string())
    return summary


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    # 1. Load
    df = load_data(INPUT_PATH)

    # 2. Preprocess
    X_scaled, scaler = scale_features(df, FEATURE_COLS)

    # 3. Train
    kmeans = fit_kmeans(X_scaled, N_CLUSTERS, RANDOM_STATE)
    df["Cluster_ID"] = kmeans.labels_

    # 4. Archetype mapping
    profiles_z = compute_cluster_profiles(kmeans.cluster_centers_, FEATURE_COLS)
    score_matrix = build_archetype_score_matrix(profiles_z)
    cluster_to_archetype = assign_archetypes(score_matrix)
    df["Archetype"] = df["Cluster_ID"].map(cluster_to_archetype)

    print("=== Cluster -> Archetype Mapping ===")
    for cluster_id, archetype in sorted(cluster_to_archetype.items()):
        print(f"  Cluster {cluster_id}  ->  {archetype}")

    # 5. Export
    df.to_csv(OUTPUT_PATH, index=False)
    joblib.dump(kmeans, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print(f"\nSaved clustered dataset -> {OUTPUT_PATH}")
    print(f"Saved model             -> {MODEL_PATH}")
    print(f"Saved scaler            -> {SCALER_PATH}")

    # Summary table for verification
    print_archetype_summary(df, FEATURE_COLS)


if __name__ == "__main__":
    main()
