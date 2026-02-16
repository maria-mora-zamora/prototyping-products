import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# GLOBAL SEED (reproducible synthetic dataset)
# -----------------------------
GLOBAL_SEED = 42
np.random.seed(GLOBAL_SEED)

DAYS_IN_MONTH = 30

st.set_page_config(page_title="Priority-Aware Budget Assistant", layout="centered")
st.title("Priority-Aware Budget Assistant")
st.caption("Prototype of an AI-assisted dynamic budgeting feature for a banking app.")


# -----------------------------
# DATA GENERATION FUNCTIONS
# -----------------------------
def simulate_transaction_history(categories_df: pd.DataFrame, n_months: int = 3) -> pd.DataFrame:
    """
    Create synthetic transaction-level history for the last n_months.
    Reproducible because of GLOBAL_SEED.
    """
    rng = np.random.default_rng(GLOBAL_SEED)
    rows = []

    for m in range(1, n_months + 1):
        for _, r in categories_df.iterrows():
            cat = str(r["category"])
            budget = float(r["budget"])
            daily_mean = budget / DAYS_IN_MONTH

            for d in range(1, DAYS_IN_MONTH + 1):
                n_tx = rng.poisson(lam=0.9)

                for _ in range(n_tx):
                    amt = rng.lognormal(mean=np.log(max(daily_mean, 1e-6)), sigma=0.6)
                    rows.append({
                        "month_idx": m,
                        "day": d,
                        "category": cat,
                        "amount": float(amt)
                    })

    return pd.DataFrame(rows)


def build_avg_cumulative_curve(history_tx: pd.DataFrame) -> pd.DataFrame:
    if history_tx.empty:
        return pd.DataFrame(columns=["category", "day", "avg_cum_frac"])

    daily = (
        history_tx.groupby(["month_idx", "category", "day"], as_index=False)["amount"].sum()
        .sort_values(["month_idx", "category", "day"])
    )

    all_months = daily["month_idx"].unique()
    all_cats = daily["category"].unique()

    full_index = pd.MultiIndex.from_product(
        [all_months, all_cats, range(1, DAYS_IN_MONTH + 1)],
        names=["month_idx", "category", "day"]
    )

    daily = daily.set_index(["month_idx", "category", "day"])\
                 .reindex(full_index, fill_value=0)\
                 .reset_index()

    daily["cum_spend"] = daily.groupby(["month_idx", "category"])["amount"].cumsum()
    monthly_total = daily.groupby(["month_idx", "category"])["amount"].transform("sum")

    daily["cum_frac"] = np.where(monthly_total > 0,
                                 daily["cum_spend"] / monthly_total,
                                 0.0)

    curve = daily.groupby(["category", "day"], as_index=False)["cum_frac"].mean()
    curve = curve.rename(columns={"cum_frac": "avg_cum_frac"})
    curve["avg_cum_frac"] = curve["avg_cum_frac"].clip(0.01, 0.99)

    return curve


def forecast_end_of_month(spent_so_far: float, day: int, avg_curve_cat: pd.DataFrame, category_name: str) -> float:
    curve_day = avg_curve_cat[
        (avg_curve_cat["category"] == category_name) &
        (avg_curve_cat["day"] == day)
    ]

    if curve_day.empty:
        return spent_so_far * (DAYS_IN_MONTH / day)

    frac = float(curve_day["avg_cum_frac"].iloc[0])
    frac = max(0.01, min(0.99, frac))

    return spent_so_far / frac


def suggest_transfers_to_target(df: pd.DataFrame, target_category: str, amount_needed: float) -> pd.DataFrame:
    if amount_needed <= 0:
        return pd.DataFrame()

    candidates = df[(df["category"] != target_category) & (df["remaining"] > 0)].copy()

    candidates = candidates.sort_values(
        by=["priority", "remaining", "budget"],
        ascending=[True, False, False]
    )

    transfers = []
    remaining_needed = amount_needed

    for _, r in candidates.iterrows():
        if remaining_needed <= 0:
            break

        max_reducible = 0.30 * float(r["budget"])
        safe_reducible = min(max_reducible, float(r["remaining"]))
        move_amount = min(safe_reducible, remaining_needed)

        if move_amount > 0:
            transfers.append({
                "from_category": r["category"],
                "from_priority": int(r["priority"]),
                "to_category": target_category,
                "amount_moved": float(move_amount),
            })
            remaining_needed -= move_amount

    return pd.DataFrame(transfers)


# -----------------------------
# 1) Budget setup
# -----------------------------
st.header("1) Budget setup")

total_budget = st.number_input("Total monthly spending budget (€)", value=1600.0)

default_categories = [
    {"category": "Groceries", "budget": 350.0, "priority": 5},
    {"category": "Eating out", "budget": 250.0, "priority": 3},
    {"category": "Leisure", "budget": 200.0, "priority": 2},
    {"category": "Transport", "budget": 100.0, "priority": 4},
]

categories = []
for i, c in enumerate(default_categories):
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        name = st.text_input("Name", value=c["category"], key=f"name_{i}")
    with col2:
        budget = st.number_input("Budget (€)", value=c["budget"], key=f"budget_{i}")
    with col3:
        priority = st.slider("Priority", 1, 5, c["priority"], key=f"prio_{i}")

    categories.append({
        "category": name,
        "budget": float(budget),
        "priority": int(priority)
    })

budgets_df = pd.DataFrame(categories)

# -----------------------------
# 2) Simulated current spending
# -----------------------------
st.header("2) Current spending (simulated)")

day = st.slider("Day of month", 1, DAYS_IN_MONTH, 15)

rng_current = np.random.default_rng(GLOBAL_SEED)

spent_list = []
for _, r in budgets_df.iterrows():
    budget = float(r["budget"])
    expected_spent = budget * 0.85 * (day / DAYS_IN_MONTH)
    simulated_spent = rng_current.normal(loc=expected_spent, scale=budget * 0.10)
    spent_list.append(max(0.0, simulated_spent))

budgets_df["spent_so_far"] = spent_list
budgets_df["remaining"] = budgets_df["budget"] - budgets_df["spent_so_far"]

st.dataframe(budgets_df, use_container_width=True)

# -----------------------------
# 3) DATA + AI Forecast
# -----------------------------
st.header("3) Data-driven Forecast (AI core)")

history_tx = simulate_transaction_history(budgets_df[["category", "budget"]])
avg_curve = build_avg_cumulative_curve(history_tx)

forecast_rows = []
for _, r in budgets_df.iterrows():
    cat = r["category"]
    spent = r["spent_so_far"]
    forecast = forecast_end_of_month(spent, day, avg_curve, cat)

    forecast_rows.append({
        "category": cat,
        "budget": r["budget"],
        "spent_so_far": spent,
        "forecast_end_month": forecast,
        "forecast_overspend_vs_budget": forecast - r["budget"]
    })

forecast_df = pd.DataFrame(forecast_rows)
st.dataframe(forecast_df, use_container_width=True)

# -----------------------------
# 4) Reallocation suggestion
# -----------------------------
st.header("4) Reallocation recommendation")

candidates = forecast_df[forecast_df["forecast_overspend_vs_budget"] > 0]

if candidates.empty:
    st.success("No category is forecasted to exceed its budget.")
else:
    target = candidates.sort_values("forecast_overspend_vs_budget", ascending=False).iloc[0]
    target_cat = target["category"]
    amount_needed = target["forecast_overspend_vs_budget"]

    st.write(f"{target_cat} is forecasted to overspend by €{amount_needed:,.0f}")

    transfers_df = suggest_transfers_to_target(budgets_df, target_cat, amount_needed)

    if transfers_df.empty:
        st.error("No safe reallocation available.")
    else:
        for _, row in transfers_df.iterrows():
            st.write(
                f"Move €{row['amount_moved']:,.0f} "
                f"from {row['from_category']} → {row['to_category']}"
            )
