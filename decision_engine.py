"""
decision_engine.py

Reusable, rule-based cart-recovery decision engine.

This module contains NO script-level side effects (no file I/O, no prints at
import time) so it can be safely imported into build_decision_engine.py,
a Streamlit app (Step 4), a notebook, or unit tests.

Core public function:
    apply_decision_engine(df, random_state=42) -> pd.DataFrame

Also exported for reuse / inspection:
    resolve_intervention(archetype, cart_value)
    compute_baseline_probability(previous_purchases, random_state=42)
    INTERVENTION_RULES
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Rule table: one row per archetype.
# `uplift` = base conversion-probability uplift from the intervention.
# `intervention` / `cost` may be a fixed value, or None if it depends on
# other row data (handled explicitly in resolve_intervention below).
# ---------------------------------------------------------------------------
INTERVENTION_RULES = {
    "Checkout Friction": {
        "intervention": "UPI Retry / Instant Payment Link",
        "uplift": 0.35,
        "cost": 2.0,
    },
    "Price Sensitive": {
        # intervention + cost depend on Cart_Value -> resolved dynamically
        "intervention": None,
        "uplift": 0.40,
        "cost": None,
        "cart_value_threshold": 800.0,
        "discount_rate": 0.10,
        "free_shipping_cost": 50.0,
    },
    "Window Shopper": {
        "intervention": "Price Drop / Low Stock Alert",
        "uplift": 0.15,
        "cost": 0.0,
    },
    "High-Intent Loyalist": {
        "intervention": "Free Express Delivery Upgrade",
        "uplift": 0.45,
        "cost": 40.0,
    },
    "Casual / Hesitant": {
        "intervention": "Standard Push Notification",
        "uplift": 0.12,
        "cost": 0.0,
    },
}


def resolve_intervention(archetype: str, cart_value: float) -> tuple[str, float, float]:
    """
    Given a user's archetype and cart value, return:
        (intervention_name, uplift, intervention_cost)

    Price Sensitive is the one archetype with conditional logic:
      - Cart_Value > 800  -> "10% Discount Coupon", cost = 10% of Cart_Value
      - Cart_Value <= 800 -> "Free Shipping",        cost = flat 50
    """
    if archetype not in INTERVENTION_RULES:
        raise ValueError(f"Unknown archetype: {archetype!r}")

    rule = INTERVENTION_RULES[archetype]

    if archetype == "Price Sensitive":
        if cart_value > rule["cart_value_threshold"]:
            intervention = "10% Discount Coupon"
            cost = round(rule["discount_rate"] * cart_value, 2)
        else:
            intervention = "Free Shipping"
            cost = rule["free_shipping_cost"]
        return intervention, rule["uplift"], cost

    return rule["intervention"], rule["uplift"], rule["cost"]


def compute_baseline_probability(previous_purchases: pd.Series, random_state: int = 42) -> np.ndarray:
    """
    Baseline (pre-intervention) recovery probability, in the 5%-15% range,
    scaled by purchase history (more repeat purchases -> slightly higher
    organic chance of coming back to finish checkout on their own), plus a
    small amount of random noise so it isn't a purely deterministic ramp.

    Purchase counts are capped at 20 for scaling purposes (users with 20+
    prior purchases are treated as equally "loyal" for this baseline).
    """
    rng = np.random.default_rng(random_state)

    capped = np.clip(previous_purchases.to_numpy(dtype=float), 0, 20)
    scaled = capped / 20.0  # 0 -> 1

    base = 0.05 + scaled * 0.10  # 5% floor, 15% ceiling before noise
    noise = rng.normal(loc=0.0, scale=0.01, size=len(previous_purchases))

    return np.clip(base + noise, 0.05, 0.15)


def apply_decision_engine(df: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """
    Run the full rule-based decision engine over a DataFrame of abandoned
    carts (must contain 'Archetype', 'Cart_Value', 'Previous_Purchases').

    Returns a NEW DataFrame (input is not mutated) with these columns added:
        Baseline_Recovery_Probability
        Recommended_Intervention
        Intervention_Cost
        Expected_Recovery_Uplift
        Post_Intervention_Probability
        Expected_Revenue_Recovered
        Net_Gain
    """
    required_cols = {"Archetype", "Cart_Value", "Previous_Purchases"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Input DataFrame is missing required columns: {missing}")

    out = df.copy()

    # --- Baseline recovery probability (independent of intervention) ------
    out["Baseline_Recovery_Probability"] = compute_baseline_probability(
        out["Previous_Purchases"], random_state=random_state
    )

    # --- Per-row intervention resolution (archetype + cart value aware) ---
    resolved = out.apply(
        lambda row: resolve_intervention(row["Archetype"], row["Cart_Value"]),
        axis=1,
        result_type="expand",
    )
    resolved.columns = ["Recommended_Intervention", "Expected_Recovery_Uplift", "Intervention_Cost"]
    out[["Recommended_Intervention", "Expected_Recovery_Uplift", "Intervention_Cost"]] = resolved

    # --- Post-intervention probability + financial metrics -----------------
    out["Post_Intervention_Probability"] = np.clip(
        out["Baseline_Recovery_Probability"] + out["Expected_Recovery_Uplift"], 0.0, 1.0
    )
    out["Expected_Revenue_Recovered"] = (
        out["Cart_Value"] * out["Post_Intervention_Probability"]
    ).round(2)
    out["Net_Gain"] = (out["Expected_Revenue_Recovered"] - out["Intervention_Cost"]).round(2)

    return out
