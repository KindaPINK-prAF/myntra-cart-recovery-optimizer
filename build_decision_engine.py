"""
build_decision_engine.py

Step 3: Apply a rule-based product decision engine to the clustered,
abandoned-cart dataset from Step 2, to decide -- per user -- which
intervention to fire, at what cost, and with what expected financial payoff.

Usage:
    python3 build_decision_engine.py

Inputs:
    myntra_carts_clustered.csv   (from Step 2)

Outputs:
    myntra_carts_decisions.csv   (final dataset with decision columns)
    Printed portfolio summary (revenue, budget, ROI, recovery improvement)

The actual decision logic lives in decision_engine.py so it can be reused
directly inside a Streamlit app in Step 4 without re-implementing anything.
"""

import pandas as pd

from decision_engine import apply_decision_engine

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_PATH = "myntra_carts_clustered.csv"
OUTPUT_PATH = "myntra_carts_decisions.csv"
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    """Load the clustered cart dataset from Step 2."""
    df = pd.read_csv(path)
    required_cols = {"Archetype", "Cart_Value", "Previous_Purchases"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in {path}: {missing}")
    return df


def print_portfolio_summary(df: pd.DataFrame) -> None:
    """Print the headline financial + recovery metrics for the whole portfolio."""
    total_abandoned_revenue = df["Cart_Value"].sum()
    baseline_revenue = (df["Cart_Value"] * df["Baseline_Recovery_Probability"]).sum()
    post_intervention_revenue = df["Expected_Revenue_Recovered"].sum()

    total_budget = df["Intervention_Cost"].sum()
    total_net_gain = df["Net_Gain"].sum()
    net_roi_pct = (total_net_gain / total_budget * 100) if total_budget > 0 else float("inf")

    # This is the metric that should land ~25-30%, matching the ~28% claim:
    # the portfolio-weighted average recovery-probability uplift delivered
    # by firing the recommended intervention on every abandoned cart.
    avg_uplift_pct = df["Expected_Recovery_Uplift"].mean() * 100

    # A separate, larger-magnitude view: how much MORE revenue is recovered
    # post-intervention relative to the "do nothing" baseline. Useful for a
    # finance-facing narrative, but this is NOT the ~28% figure.
    relative_revenue_uplift_pct = (
        (post_intervention_revenue - baseline_revenue) / baseline_revenue * 100
    )

    print("=" * 66)
    print("PORTFOLIO SUMMARY")
    print("=" * 66)
    print(f"Total Abandoned Revenue (Cart_Value sum) : Rs {total_abandoned_revenue:,.2f}")
    print(f"Baseline Recovery Revenue (no action)     : Rs {baseline_revenue:,.2f}")
    print(f"Post-Intervention Revenue                 : Rs {post_intervention_revenue:,.2f}")
    print("-" * 66)
    print(f"Avg. Recovery-Probability Uplift (pp)     : {avg_uplift_pct:.2f}%  "
          f"<-- headline ~28% recovery-improvement figure")
    print(f"Relative Revenue Improvement vs Baseline   : {relative_revenue_uplift_pct:.2f}%  "
          f"(revenue-dollar view, naturally larger)")
    print("-" * 66)
    print(f"Total Intervention Budget                 : Rs {total_budget:,.2f}")
    print(f"Total Net Gain (Revenue - Cost)            : Rs {total_net_gain:,.2f}")
    print(f"Net ROI on Intervention Spend               : {net_roi_pct:,.1f}%")
    print("=" * 66)

    # Sanity check against the ~28% claim
    if 25 <= avg_uplift_pct <= 30:
        print(f"[OK] Avg. recovery-probability uplift ({avg_uplift_pct:.1f}%) "
              f"falls within the ~25-30% target band.")
    else:
        print(f"[NOTE] Avg. recovery-probability uplift ({avg_uplift_pct:.1f}%) "
              f"is OUTSIDE the ~25-30% target band -- check archetype mix.")

    # Per-archetype breakdown, useful for a resume bullet / stakeholder deck
    print("\nPer-Archetype Breakdown:")
    breakdown = (
        df.groupby(["Archetype", "Recommended_Intervention"])
        .agg(
            Users=("Cart_Value", "size"),
            Avg_Cart_Value=("Cart_Value", "mean"),
            Avg_Uplift_Pct=("Expected_Recovery_Uplift", lambda x: x.mean() * 100),
            Total_Cost=("Intervention_Cost", "sum"),
            Total_Net_Gain=("Net_Gain", "sum"),
        )
        .round(2)
        .sort_values("Total_Net_Gain", ascending=False)
    )
    print(breakdown.to_string())


def main():
    df = load_data(INPUT_PATH)

    df_decisions = apply_decision_engine(df, random_state=RANDOM_STATE)

    df_decisions.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved decision-engine output -> {OUTPUT_PATH}\n")

    print_portfolio_summary(df_decisions)


if __name__ == "__main__":
    main()
