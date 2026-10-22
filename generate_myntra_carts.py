"""
generate_myntra_carts.py

Generates a synthetic dataset of 10,000 abandoned e-commerce cart records
and exports it to 'myntra_carts.csv'.

Columns:
- User_ID              : unique identifier for each user/session
- Cart_Value           : total value (INR) of items left in the cart
- Time_Spent_In_App    : minutes spent in-app during the session before abandonment
- Payment_Failures     : number of failed payment attempts during checkout
- Previous_Purchases   : number of prior completed purchases by this user
- Discount_Sensitivity : score from 0 (indifferent to discounts) to 1 (highly discount-driven)
"""

import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

N = 10_000

# ---------------------------------------------------------------------------
# User_ID: simple unique identifiers
# ---------------------------------------------------------------------------
user_ids = [f"USER_{i:06d}" for i in range(1, N + 1)]

# ---------------------------------------------------------------------------
# Previous_Purchases: most users have few, some are frequent repeat buyers
# (right-skewed, non-negative integers)
# ---------------------------------------------------------------------------
previous_purchases = np.random.negative_binomial(n=2, p=0.35, size=N)
previous_purchases = np.clip(previous_purchases, 0, 60)

# ---------------------------------------------------------------------------
# Discount_Sensitivity: score in [0, 1]. New/low-purchase users tend to skew
# slightly higher on average (more price-conscious first-time browsers),
# but with plenty of noise so it isn't a trivial derived feature.
# ---------------------------------------------------------------------------
base_sensitivity = np.random.beta(a=2.2, b=2.0, size=N)
purchase_adj = -0.01 * np.minimum(previous_purchases, 20)  # loyal buyers slightly less discount-driven
discount_sensitivity = np.clip(base_sensitivity + purchase_adj + np.random.normal(0, 0.05, N), 0, 1)
discount_sensitivity = np.round(discount_sensitivity, 3)

# ---------------------------------------------------------------------------
# Cart_Value: lognormal distribution (INR), typical fashion e-commerce range
# roughly 300 - 20,000, with a long tail for bulk/high-value carts
# ---------------------------------------------------------------------------
cart_value = np.random.lognormal(mean=7.2, sigma=0.6, size=N)
cart_value = np.clip(cart_value, 250, 25000)
cart_value = np.round(cart_value, 2)

# ---------------------------------------------------------------------------
# Time_Spent_In_App: minutes spent in the session before abandoning.
# Gamma distribution: most sessions short, some long browsing sessions.
# ---------------------------------------------------------------------------
time_spent = np.random.gamma(shape=2.5, scale=3.2, size=N)
time_spent = np.clip(time_spent, 0.5, 90)
time_spent = np.round(time_spent, 2)

# ---------------------------------------------------------------------------
# Payment_Failures: most carts abandon with 0 failures (never reached
# checkout / gave up before paying); a meaningful subset abandon *because*
# of failed payment attempts.
# ---------------------------------------------------------------------------
payment_failures = np.random.choice(
    [0, 1, 2, 3, 4],
    size=N,
    p=[0.62, 0.20, 0.11, 0.05, 0.02]
)

# ---------------------------------------------------------------------------
# Assemble DataFrame
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "User_ID": user_ids,
    "Cart_Value": cart_value,
    "Time_Spent_In_App": time_spent,
    "Payment_Failures": payment_failures,
    "Previous_Purchases": previous_purchases,
    "Discount_Sensitivity": discount_sensitivity,
})

# Shuffle rows so column-generation order doesn't leave any accidental ordering
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

output_path = "myntra_carts.csv"
df.to_csv(output_path, index=False)

print(f"Generated {len(df):,} rows -> {output_path}")
print(df.describe(include="all"))
