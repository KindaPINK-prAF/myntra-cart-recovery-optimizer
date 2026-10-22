"""
app.py

Step 4: Streamlit front-end for the Myntra Cart Recovery Optimizer.

Reuses the exact rule-based decision engine from Step 3 (decision_engine.py)
so nothing is re-implemented here -- this app is a *view* on top of the
same logic that produced myntra_carts_decisions.csv.

Run with:
    streamlit run app.py
"""

import math

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy.stats import norm

from decision_engine import apply_decision_engine

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Myntra Cart Recovery Optimizer",
    page_icon="🛒",
    layout="wide",
)

DATA_PATH = "myntra_carts_decisions.csv"


# ---------------------------------------------------------------------------
# Data loading (cached so the CSV is only read once per session)
# ---------------------------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@st.cache_data
def portfolio_summary(df: pd.DataFrame) -> dict:
    """Headline numbers for the Stakeholder Dashboard."""
    total_abandoned_revenue = df["Cart_Value"].sum()
    baseline_revenue = (df["Cart_Value"] * df["Baseline_Recovery_Probability"]).sum()
    post_intervention_revenue = df["Expected_Revenue_Recovered"].sum()
    total_budget = df["Intervention_Cost"].sum()
    total_net_gain = df["Net_Gain"].sum()
    avg_uplift_pct = df["Expected_Recovery_Uplift"].mean() * 100

    return {
        "total_abandoned_revenue": total_abandoned_revenue,
        "baseline_revenue": baseline_revenue,
        "post_intervention_revenue": post_intervention_revenue,
        "total_budget": total_budget,
        "total_net_gain": total_net_gain,
        "avg_uplift_pct": avg_uplift_pct,
    }


try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(
        f"Could not find `{DATA_PATH}`. Run Steps 1-3 first "
        "(generate_myntra_carts.py -> cluster_myntra_carts.py -> build_decision_engine.py) "
        "so this file exists in the same folder as app.py."
    )
    st.stop()

summary = portfolio_summary(df)

ARCHETYPE_DESCRIPTIONS = {
    "Checkout Friction": "Tried to pay, failed, and gave up. The cart is a technical problem, not a motivation problem.",
    "Price Sensitive": "Ready to buy, but the price (or the shipping fee) is the sticking point.",
    "Window Shopper": "Spent a long time browsing but never built up a high-value cart -- still exploring.",
    "High-Intent Loyalist": "Repeat customers with high-value carts -- your closest-to-conversion, highest-value users.",
    "Casual / Hesitant": "Low engagement across the board -- browsing casually, no strong signal either way.",
}


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🛒 Cart Recovery Optimizer")
st.sidebar.caption("Myntra-inspired cart abandonment case study")
st.sidebar.divider()

section = st.sidebar.radio(
    "Navigate to:",
    [
        "📊 Stakeholder Dashboard",
        "🎯 Single-User Optimizer",
        "🧪 A/B Test Planner",
        "📖 Methodology",
    ],
)

st.sidebar.divider()
st.sidebar.caption(f"Dataset: {len(df):,} abandoned carts loaded")


# ---------------------------------------------------------------------------
# SECTION 1: Stakeholder Dashboard
# ---------------------------------------------------------------------------
if section == "📊 Stakeholder Dashboard":
    st.title("📊 Stakeholder Dashboard")
    st.caption("High-level portfolio view of the cart recovery opportunity")

    st.success(
        f"**Projected recovery-probability uplift: {summary['avg_uplift_pct']:.1f}%** "
        f"across {len(df):,} abandoned carts, driven by 5 archetype-targeted interventions."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Abandoned Revenue", f"₹{summary['total_abandoned_revenue']:,.0f}")
    col2.metric("Projected Recovery Uplift", f"{summary['avg_uplift_pct']:.1f}%")
    col3.metric("Total Intervention Budget", f"₹{summary['total_budget']:,.0f}")
    col4.metric(
        "Total Net Gain",
        f"₹{summary['total_net_gain']:,.0f}",
        help="Expected recovered revenue minus total intervention cost",
    )

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.subheader("Archetype Distribution")
        archetype_counts = df["Archetype"].value_counts().reset_index()
        archetype_counts.columns = ["Archetype", "Users"]
        fig_pie = px.pie(
            archetype_counts,
            names="Archetype",
            values="Users",
            hole=0.45,
        )
        fig_pie.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.subheader("Expected Revenue Recovered by Archetype")
        revenue_by_archetype = (
            df.groupby("Archetype")["Expected_Revenue_Recovered"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig_bar = px.bar(
            revenue_by_archetype,
            x="Archetype",
            y="Expected_Revenue_Recovered",
            color="Archetype",
            text_auto=".2s",
            labels={"Expected_Revenue_Recovered": "Expected Revenue Recovered (₹)"},
        )
        fig_bar.update_layout(showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.info(
        "**Reading this dashboard:** the recovery-probability uplift is the metric that "
        "matches the resume claim (~28%) -- it's the portfolio-weighted average improvement "
        "in checkout probability delivered by firing the right intervention on every cart. "
        "Net Gain is the dollar-value takeaway for a stakeholder deck."
    )


# ---------------------------------------------------------------------------
# SECTION 2: Single-User Optimizer
# ---------------------------------------------------------------------------
elif section == "🎯 Single-User Optimizer":
    st.title("🎯 Single-User Optimizer")
    st.caption("Look up any abandoned cart and see the recommended intervention")

    selected_user = st.selectbox("Select a User_ID", options=df["User_ID"].tolist())
    row = df.loc[df["User_ID"] == selected_user].iloc[0]

    st.divider()
    st.subheader(f"Profile: {selected_user}")

    p1, p2, p3 = st.columns(3)
    p1.metric("Cart Value", f"₹{row['Cart_Value']:,.2f}")
    p1.metric("Time Spent In App", f"{row['Time_Spent_In_App']:.1f} min")
    p2.metric("Payment Failures", int(row["Payment_Failures"]))
    p2.metric("Previous Purchases", int(row["Previous_Purchases"]))
    p3.metric("Discount Sensitivity", f"{row['Discount_Sensitivity']:.2f}")
    p3.metric("Archetype", row["Archetype"])

    st.divider()
    st.subheader("Recommended Action")

    st.success(f"**{row['Recommended_Intervention']}**")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Intervention Cost", f"₹{row['Intervention_Cost']:,.2f}")
    m2.metric("Recovery Uplift", f"+{row['Expected_Recovery_Uplift'] * 100:.0f} pp")
    m3.metric("Post-Intervention Probability", f"{row['Post_Intervention_Probability'] * 100:.1f}%")
    m4.metric("Expected Revenue Recovered", f"₹{row['Expected_Revenue_Recovered']:,.2f}")

    if row["Intervention_Cost"] > 0:
        user_roi_pct = row["Net_Gain"] / row["Intervention_Cost"] * 100
        st.metric("Net Gain (this user)", f"₹{row['Net_Gain']:,.2f}", delta=f"{user_roi_pct:,.0f}% ROI")
    else:
        st.metric("Net Gain (this user)", f"₹{row['Net_Gain']:,.2f}", delta="Zero-cost intervention (ROI: N/A)")

    st.divider()

    with st.expander("🔍 What-if: simulate a different cart value for this user"):
        st.caption(
            "Uses the same `apply_decision_engine()` function from decision_engine.py -- "
            "no logic is duplicated here, this just re-runs the real engine on a hypothetical cart value."
        )
        hypothetical_value = st.slider(
            "Hypothetical Cart Value (₹)",
            min_value=100.0,
            max_value=25000.0,
            value=float(row["Cart_Value"]),
            step=50.0,
        )
        what_if_row = pd.DataFrame([{
            "Archetype": row["Archetype"],
            "Cart_Value": hypothetical_value,
            "Previous_Purchases": row["Previous_Purchases"],
        }])
        what_if_result = apply_decision_engine(what_if_row, random_state=42).iloc[0]

        w1, w2, w3 = st.columns(3)
        w1.metric("Intervention", what_if_result["Recommended_Intervention"])
        w2.metric("New Cost", f"₹{what_if_result['Intervention_Cost']:,.2f}")
        w3.metric("New Expected Revenue Recovered", f"₹{what_if_result['Expected_Revenue_Recovered']:,.2f}")


# ---------------------------------------------------------------------------
# SECTION 3: A/B Test Planner
# ---------------------------------------------------------------------------
elif section == "🧪 A/B Test Planner":
    st.title("🧪 A/B Test Planner")
    st.caption("Sample-size calculator for testing a new recovery intervention")

    st.info(
        "Assumes a standard two-sided test at **95% confidence** and **80% power** "
        "(industry-default settings for most product experiments)."
    )

    s1, s2, s3 = st.columns(3)
    with s1:
        baseline_conversion_pct = st.slider(
            "Baseline Conversion Rate (%)", min_value=1.0, max_value=50.0, value=10.0, step=0.5
        )
    with s2:
        mde_pct = st.slider(
            "Minimum Detectable Effect (percentage points)",
            min_value=0.5,
            max_value=20.0,
            value=2.0,
            step=0.5,
        )
    with s3:
        daily_traffic = st.slider(
            "Daily Traffic (carts/day, split across both variants)",
            min_value=50,
            max_value=20000,
            value=1000,
            step=50,
        )

    alpha = 0.05
    power = 0.80

    p1_conv = baseline_conversion_pct / 100
    p2_conv = p1_conv + (mde_pct / 100)
    p2_conv = min(p2_conv, 0.999)  # guard against >100% conversion

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(power)

    pooled_p = (p1_conv + p2_conv) / 2
    numerator = (
        z_alpha * math.sqrt(2 * pooled_p * (1 - pooled_p))
        + z_beta * math.sqrt(p1_conv * (1 - p1_conv) + p2_conv * (1 - p2_conv))
    ) ** 2
    denominator = (p2_conv - p1_conv) ** 2

    n_per_variant = math.ceil(numerator / denominator) if denominator > 0 else float("inf")
    total_sample = n_per_variant * 2
    days_needed = math.ceil(total_sample / daily_traffic) if daily_traffic > 0 else float("inf")

    st.divider()

    r1, r2, r3 = st.columns(3)
    r1.metric("Sample Size (per variant)", f"{n_per_variant:,}")
    r2.metric("Total Sample Size", f"{total_sample:,}")
    r3.metric("Estimated Test Duration", f"{days_needed:,} days")

    st.divider()

    st.success(
        f"To detect a lift from **{baseline_conversion_pct:.1f}%** to "
        f"**{p2_conv * 100:.1f}%** conversion (a {mde_pct:.1f} pp improvement) with "
        f"95% confidence and 80% power, you'll need **{total_sample:,} total carts** "
        f"(**{n_per_variant:,} per variant**), which takes about **{days_needed} days** "
        f"at {daily_traffic:,} carts/day."
    )

    with st.expander("📈 How sample size changes with the effect size you're chasing"):
        mde_range = np.arange(0.5, 20.5, 0.5)
        sample_sizes = []
        for mde in mde_range:
            p2_i = min(p1_conv + mde / 100, 0.999)
            pooled_i = (p1_conv + p2_i) / 2
            denom_i = (p2_i - p1_conv) ** 2
            if denom_i == 0:
                sample_sizes.append(np.nan)
                continue
            num_i = (
                z_alpha * math.sqrt(2 * pooled_i * (1 - pooled_i))
                + z_beta * math.sqrt(p1_conv * (1 - p1_conv) + p2_i * (1 - p2_i))
            ) ** 2
            sample_sizes.append(math.ceil(num_i / denom_i) * 2)

        sensitivity_df = pd.DataFrame({
            "Minimum Detectable Effect (pp)": mde_range,
            "Total Sample Size Needed": sample_sizes,
        })
        fig_sensitivity = px.line(
            sensitivity_df,
            x="Minimum Detectable Effect (pp)",
            y="Total Sample Size Needed",
            markers=False,
        )
        st.plotly_chart(fig_sensitivity, use_container_width=True)
        st.caption(
            "Smaller effects require dramatically larger samples -- useful for setting "
            "expectations with stakeholders about test duration before committing to an MDE."
        )


# ---------------------------------------------------------------------------
# SECTION 4: Methodology
# ---------------------------------------------------------------------------
elif section == "📖 Methodology":
    st.title("📖 Methodology")
    st.caption("How this recommendation engine works, explained for a non-technical audience")

    st.markdown(
        """
### The problem
Not every abandoned cart is abandoned for the same reason. A one-size-fits-all
"come back!" email ignores that some users left because payment failed, others
because the price felt too high, and others were just browsing. Treating them
differently should recover more revenue for the same (or lower) cost.
"""
    )

    st.divider()
    st.subheader("Step 1-2: Finding natural customer groups (K-Means clustering)")
    st.markdown(
        """
We don't manually decide who belongs in which group -- we let the data find its
own patterns. Each abandoned cart is described by 5 numbers: cart value, time
spent in the app, number of payment failures, number of past purchases, and a
discount-sensitivity score.

**K-Means** is an algorithm that looks at all 10,000 carts in that 5-number
"space" and finds 5 natural groupings, where carts inside the same group behave
similarly to each other and differently from carts in other groups. It doesn't
know anything about *why* the groups differ -- it just finds where the natural
clusters of similar behavior sit.

Once we have 5 clusters, we look at each cluster's average behavior (e.g. "this
cluster has unusually high payment failures") and assign it a plain-English name
automatically, using an optimization method that matches each cluster to the
label that fits it best -- without any group accidentally getting the same label
twice.
"""
    )

    st.divider()
    st.subheader("The 5 archetypes")

    for archetype, description in ARCHETYPE_DESCRIPTIONS.items():
        with st.expander(f"**{archetype}**"):
            st.write(description)
            example_row = df[df["Archetype"] == archetype].iloc[0]
            st.caption(
                f"Recommended intervention: **{example_row['Recommended_Intervention']}** "
                f"(cost ₹{example_row['Intervention_Cost']:,.2f}, "
                f"+{example_row['Expected_Recovery_Uplift'] * 100:.0f} pp recovery uplift)"
            )

    st.divider()
    st.subheader("Step 3: Turning archetypes into decisions")
    st.markdown(
        """
Each archetype gets one specific intervention, matched to *why* that group likely
abandoned:

- **Checkout Friction** → instant payment retry link (the problem was technical, not motivational)
- **Price Sensitive** → a discount or free shipping (the problem was cost)
- **Window Shopper** → a low-stock nudge (the problem was low urgency)
- **High-Intent Loyalist** → a free delivery upgrade (a small, low-cost nudge for an already-likely buyer)
- **Casual / Hesitant** → a simple reminder push notification (low investment, low expected return)

For every user, we estimate a **baseline** chance they'd complete the purchase
with no intervention at all, then add the intervention's expected uplift to get
a **post-intervention** probability. Multiplying that by the cart value gives
expected recovered revenue; subtracting the intervention's cost gives the net
financial gain.
"""
    )

    st.divider()
    st.info(
        "**Caveat:** this project uses a synthetic dataset built to resemble realistic "
        "e-commerce behavior, not real Myntra transaction data. The pipeline, methodology, "
        "and decision logic are the actual deliverable -- swap in real data and the same "
        "code runs unchanged."
    )
