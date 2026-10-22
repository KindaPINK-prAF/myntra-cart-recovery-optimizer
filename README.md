# 🛒 Myntra Cart Recovery Optimizer

A data-driven decision engine that segments abandoned e-commerce carts into behavioral archetypes and recommends a targeted, cost-aware intervention for each one. It ships with a Streamlit app for stakeholders and product managers.

> **Live app:** [Open the Streamlit app](https://myntra-cart-recovery-optimizer-sdwgt7l25th2kmwyxwmmf2.streamlit.app/)

---

## The Problem

Most abandoned-cart campaigns send every user the same generic reminder. But users abandon for different reasons: a payment failed, the price felt too high, or they were just browsing. Matching the intervention to the *reason* should recover more revenue for the same or lower spend.

## Approach

The project is a 4-step pipeline:

| Step | Script | What it does |
|------|--------|--------------|
| 1 | `generate_myntra_carts.py` | Generates a synthetic dataset of 10,000 abandoned carts |
| 2 | `cluster_myntra_carts.py` | Scales features and runs K-Means (k=5), then maps clusters to named archetypes using the Hungarian algorithm |
| 3 | `build_decision_engine.py` + `decision_engine.py` | Applies rule-based interventions and computes recovery probability, expected revenue and net gain |
| 4 | `app.py` | Streamlit app with a stakeholder dashboard, single-user optimizer, A/B test planner and methodology page |

### The 5 Archetypes

| Archetype | Signal | Intervention | Uplift | Cost |
|-----------|--------|--------------|--------|------|
| Checkout Friction | High payment failures | UPI Retry / Instant Payment Link | +35% | ₹2 |
| Price Sensitive | High discount sensitivity | 10% Coupon (cart > ₹800) or Free Shipping | +40% | 10% of cart, or ₹50 |
| Window Shopper | Long time in app, lower cart value | Price Drop / Low Stock Alert | +15% | ₹0 |
| High-Intent Loyalist | Many past purchases, high cart value | Free Express Delivery Upgrade | +45% | ₹40 |
| Casual / Hesitant | Low purchase history, low activity | Standard Push Notification | +12% | ₹0 |

## Key Results (projected, on synthetic data)

- **Total abandoned revenue:** ₹1.61 crore across 10,000 carts
- **Average recovery-probability uplift:** **28.5%**, the archetype uplifts weighted by the cluster mix
- **Total intervention budget:** ~₹4.86 lakh
- **Total net gain (expected recovered revenue minus intervention cost):** ~₹55.9 lakh

## App Features

- 📊 **Stakeholder Dashboard:** headline metrics, archetype distribution and expected revenue by archetype
- 🎯 **Single-User Optimizer:** look up any `User_ID` to see their archetype, recommended intervention, cost and ROI, plus a what-if simulator
- 🧪 **A/B Test Planner:** sample-size and test-duration calculator (95% confidence, 80% power)
- 📖 **Methodology:** plain-English explainer of the clustering and decision logic

## Run Locally

```bash
git clone https://github.com/KindaPINK-prAF/myntra-cart-recovery-optimizer.git
cd myntra-cart-recovery-optimizer
pip install -r requirements.txt
streamlit run app.py
```

To regenerate everything from scratch:

```bash
python generate_myntra_carts.py
python cluster_myntra_carts.py
python build_decision_engine.py
streamlit run app.py
```

## Project Structure

```
├── app.py                      # Streamlit front-end
├── decision_engine.py          # Reusable decision logic (imported by the app)
├── generate_myntra_carts.py    # Step 1: synthetic data
├── cluster_myntra_carts.py     # Step 2: K-Means + archetype mapping
├── build_decision_engine.py    # Step 3: apply decisions, portfolio summary
├── myntra_carts.csv            # Raw synthetic carts
├── myntra_carts_clustered.csv  # With Cluster_ID and Archetype
├── myntra_carts_decisions.csv  # With interventions and financial metrics
├── kmeans_model.pkl            # Fitted K-Means model
├── scaler.pkl                  # Fitted StandardScaler
└── requirements.txt
```

## Assumptions & Limitations

- **The data is synthetic.** It was generated to resemble realistic e-commerce behavior and is not real Myntra data. The pipeline and decision logic are the deliverable, and real data can be dropped in.
- **Uplift figures are assumptions, not measured results.** The 28.5% figure is a *projected* uplift, built from assumed per-archetype uplifts weighted by cluster size. It would need to be validated with a real A/B test, which is what the A/B Test Planner is for.
- Baseline recovery probability (5–15%) is modeled from purchase history with a small amount of noise.

## Tech Stack

Python · Pandas · NumPy · scikit-learn · SciPy · Plotly · Streamlit