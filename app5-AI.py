import streamlit as st
import pandas as pd
import numpy as np

# -----------------------------
# Global config (reproducible data)
# -----------------------------
GLOBAL_SEED = 42
DAYS_IN_MONTH = 30

st.set_page_config(page_title="Priority-Aware Budget Assistant", layout="centered")
st.title("Priority-Aware Budget Assistant")
st.caption("Prototype of an AI-assisted dynamic budgeting feature for a banking app.")


# -----------------------------
# Data + AI-lite helpers
# -----------------------------
def simulate_transaction_history(categories_df: pd.DataFrame, n_months: int = 3, seed: int = GLOBAL_SEED) -> pd.DataFrame:
    """
    Create synthetic transaction-level history for the last n_months.
    Output columns: month_idx, day, category, amount
    """
    rng = np.random.default_rng(seed)
    rows = []

    for m in range(1, n_months + 1):
        for _, r in categories_df.iterrows():
            cat = str(r["category"])
            budget = float(r["budget"])
            daily_mean = budget / DAYS_IN_MONTH

            for d in range(1, DAYS_IN_MONTH + 1):
                # typical number of transactions per day
                n_tx = rng.poisson(lam=0.9)
                for _ in range(n_tx):
                    # lognormal → small tx most days, occasional bigger tx
                    amt = rng.lognormal(mean=np.log(max(daily_mean, 1e-6)), sigma=0.6)
                    rows.append({"month_idx": m, "day": d, "category": cat, "amount": float(amt)})

    return pd.DataFrame(rows)


def build_avg_cumulative_curve(history_tx: pd.DataFrame) -> pd.DataFrame:
    """
    Compute average cumulative spend fraction curve per day, per category.
    Output: category, day, avg_cum_frac (clipped to [0.01, 0.99])
    """
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
        names=["month_idx", "category", "day"],
    )

    daily = (
        daily.set_index(["month_idx", "category", "day"])
        .reindex(full_index, fill_value=0)
        .reset_index()
    )

    daily["cum_spend"] = daily.groupby(["month_idx", "category"])["amount"].cumsum()
    monthly_total = daily.groupby(["month_idx", "category"])["amount"].transform("sum")

    daily["cum_frac"] = np.where(monthly_total > 0, daily["cum_spend"] / monthly_total, 0.0)

    curve = daily.groupby(["category", "day"], as_index=False)["cum_frac"].mean()
    curve = curve.rename(columns={"cum_frac": "avg_cum_frac"})
    curve["avg_cum_frac"] = curve["avg_cum_frac"].clip(lower=0.01, upper=0.99)

    return curve


def forecast_end_of_month(spent_so_far: float, day: int, avg_curve: pd.DataFrame, category_name: str) -> float:
    """
    Forecast EOM spend using historical cumulative fraction curve:
        forecast = spent_so_far / avg_cum_frac(day)
    Fallback: simple pace if curve missing.
    """
    curve_day = avg_curve[(avg_curve["category"] == category_name) & (avg_curve["day"] == day)]
    if curve_day.empty:
        return spent_so_far * (DAYS_IN_MONTH / max(day, 1))

    frac = float(curve_day["avg_cum_frac"].iloc[0])
    frac = max(0.01, min(0.99, frac))
    return spent_so_far / frac


def suggest_transfers_to_target(df: pd.DataFrame, target_category: str, amount_needed: float) -> pd.DataFrame:
    """
    Suggest transfers FROM other categories TO the target category.
    - lowest priority first
    - only from categories with remaining budget
    - cap at 30% of source budget
    - never reduce below already spent (ensured via remaining)
    """
    if amount_needed <= 0:
        return pd.DataFrame()

    candidates = df[(df["category"] != target_category) & (df["remaining"] > 0)].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates = candidates.sort_values(by=["priority", "remaining", "budget"], ascending=[True, False, False])

    transfers = []
    remaining_needed = amount_needed

    for _, r in candidates.iterrows():
        if remaining_needed <= 0:
            break

        max_reducible = 0.30 * float(r["budget"])
        safe_reducible = min(max_reducible, float(r["remaining"]))
        move_amount = min(safe_reducible, remaining_needed)

        if move_amount > 0:
            transfers.append(
                {
                    "from_category": str(r["category"]),
                    "from_priority": int(r["priority"]),
                    "to_category": target_category,
                    "amount_moved": float(move_amount),
                }
            )
            remaining_needed -= move_amount

    out = pd.DataFrame(transfers)
    if not out.empty:
        out.attrs["uncovered_amount"] = float(max(0.0, remaining_needed))
    return out


def highlight_gap(val):
    # overspend positive → red; surplus negative/zero → green
    try:
        v = float(val)
    except Exception:
        return ""
    if v > 0:
        return "color: red; font-weight: 700;"
    return "color: green;"


# -----------------------------
# 1) Budget setup
# -----------------------------
st.header("1) Budget setup")

col1, col2 = st.columns(2)
with col1:
    monthly_income = st.number_input("Monthly income (€)", min_value=0.0, value=2500.0, step=50.0)
with col2:
    total_budget = st.number_input("Total monthly spending budget (€)", min_value=0.0, value=1600.0, step=50.0)

st.markdown("Define category budgets and priorities (1 = low priority, 5 = high priority).")

default_categories = [
    {"category": "Groceries", "budget": 350.0, "priority": 5},
    {"category": "Eating out", "budget": 250.0, "priority": 3},
    {"category": "Leisure", "budget": 200.0, "priority": 2},
    {"category": "Transport", "budget": 100.0, "priority": 4},
]

n_categories = st.slider("Number of categories", min_value=2, max_value=6, value=4)

categories = []
for i in range(n_categories):
    st.subheader(f"Category {i+1}")

    default_name = default_categories[i]["category"] if i < len(default_categories) else f"Category {i+1}"
    default_budget = default_categories[i]["budget"] if i < len(default_categories) else 100.0
    default_priority = default_categories[i]["priority"] if i < len(default_categories) else 3

    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        name = st.text_input("Name", value=default_name, key=f"name_{i}")
    with c2:
        budget = st.number_input("Budget (€)", min_value=0.0, value=float(default_budget), step=10.0, key=f"budget_{i}")
    with c3:
        priority = st.slider("Priority", 1, 5, int(default_priority), key=f"prio_{i}")

    categories.append({"category": name.strip(), "budget": float(budget), "priority": int(priority)})

categories = [c for c in categories if c["category"] != ""]
if len(categories) == 0:
    st.error("Please add at least one category (category name cannot be empty).")
    st.stop()

budgets_df = pd.DataFrame(categories)

# Allocation check (informative)
total_planned = float(budgets_df["budget"].sum())
allocation_gap = float(total_budget - total_planned)

st.subheader("Allocation check")

cA, cB = st.columns(2)
cA.metric("Total allocated across categories (€)", f"{total_planned:,.0f}")
cB.metric("Remaining to allocate (€)", f"{allocation_gap:,.0f}")

if allocation_gap > 0:
    st.info(f"You still have **€{allocation_gap:,.0f}** unallocated.")
elif allocation_gap < 0:
    st.error(f"You are **€{abs(allocation_gap):,.0f}** over budget (categories sum above total).")
else:
    st.success("Perfect! Categories add up exactly to your total monthly budget.")

if total_budget > 0:
    pct = min(total_planned / total_budget, 1.0)
    st.progress(pct, text=f"Allocated {total_planned:,.0f} / {total_budget:,.0f} (€)")

st.divider()


# -----------------------------
# 2) Current spending (manual OR simulated)
# -----------------------------
st.header("2) Current spending (manual or simulated)")

manual_mode = st.checkbox("Manual mode: enter current spending yourself")

if manual_mode:
    day = st.slider("Day of month", 1, DAYS_IN_MONTH, 15)
    spending = []
    for i, row in budgets_df.iterrows():
        spent = st.number_input(
            f"Current spent in {row['category']} (€)",
            min_value=0.0,
            value=0.0,
            step=10.0,
            key=f"manual_spent_{i}",
        )
        spending.append(float(spent))
else:
    st.write("Simulated spending up to a given day (reproducible with a fixed seed).")

    colS1, colS2 = st.columns([1, 1])
    with colS1:
        day = st.slider("Day of month", 1, DAYS_IN_MONTH, 15)
    with colS2:
        style = st.selectbox("Spending style", ["Cautious", "Typical", "Impulsive"], index=1)

    style_params = {
        "Cautious": {"mean_multiplier": 0.70, "noise_multiplier": 0.08},
        "Typical": {"mean_multiplier": 0.85, "noise_multiplier": 0.10},
        "Impulsive": {"mean_multiplier": 1.00, "noise_multiplier": 0.14},
    }
    mean_mult = style_params[style]["mean_multiplier"]
    noise_mult = style_params[style]["noise_multiplier"]

    # fixed seed so the same inputs produce the same simulated spending
    rng_current = np.random.default_rng(GLOBAL_SEED)

    spending = []
    for _, r in budgets_df.iterrows():
        budget = float(r["budget"])
        expected_spent = budget * mean_mult * (day / DAYS_IN_MONTH)
        simulated_spent = rng_current.normal(loc=expected_spent, scale=budget * noise_mult)
        spending.append(float(max(0.0, simulated_spent)))

# common computed columns
budgets_df["spent_so_far"] = spending
budgets_df["remaining"] = budgets_df["budget"] - budgets_df["spent_so_far"]

st.subheader("Spending snapshot")
st.dataframe(budgets_df[["category", "budget", "priority", "spent_so_far", "remaining"]], use_container_width=True)

total_spent = float(budgets_df["spent_so_far"].sum())
st.metric("Total spent so far (€)", f"{total_spent:,.0f}")

st.divider()


# -----------------------------
# 3) Data-driven Forecast (AI core)
# -----------------------------
st.header("3) Data-driven Forecast (AI core)")

st.write(
    "The prototype generates a **synthetic but reproducible** transaction history and uses it to forecast end-of-month spending."
)

history_tx = simulate_transaction_history(budgets_df[["category", "budget"]], n_months=3, seed=GLOBAL_SEED)
avg_curve = build_avg_cumulative_curve(history_tx)

with st.expander("See sample synthetic transactions (data)"):
    st.dataframe(history_tx.head(20), use_container_width=True)

forecast_rows = []
for _, r in budgets_df.iterrows():
    cat = str(r["category"])
    spent = float(r["spent_so_far"])
    budget = float(r["budget"])
    forecast = float(forecast_end_of_month(spent, day, avg_curve, cat))
    gap = forecast - budget  # >0 overspend, <0 surplus

    forecast_rows.append(
        {
            "category": cat,
            "budget": budget,
            "spent_so_far": spent,
            "forecast_end_month": forecast,
            "forecast_overspend_vs_budget": gap,
        }
    )

forecast_df = pd.DataFrame(forecast_rows)

# Styled forecast table (red overspend, green surplus)
styled_forecast = forecast_df.style.applymap(highlight_gap, subset=["forecast_overspend_vs_budget"])

st.dataframe(
    styled_forecast,
    use_container_width=True
)

forecast_total = float(forecast_df["forecast_end_month"].sum())
st.metric("Forecast total end-of-month spend (€)", f"{forecast_total:,.0f}")

st.divider()


# -----------------------------
# 4) Reallocation recommendation (driven by forecast)
# -----------------------------
st.header("4) Reallocation recommendation")

candidates = forecast_df[forecast_df["forecast_overspend_vs_budget"] > 0].copy()

if candidates.empty:
    st.success("No category is forecasted to exceed its budget. No reallocation needed.")
else:
    target = candidates.sort_values("forecast_overspend_vs_budget", ascending=False).iloc[0]
    target_cat = str(target["category"])
    amount_needed = float(target["forecast_overspend_vs_budget"])

    st.write(f"**{target_cat}** is forecasted to overspend by **€{amount_needed:,.0f}**.")

    transfers_df = suggest_transfers_to_target(budgets_df, target_cat, amount_needed)

    if transfers_df.empty:
        st.error("No safe reallocation available (no remaining budget in other categories).")
    else:
        # easy-to-read suggestions
        remaining_map = budgets_df.set_index("category")["remaining"].to_dict()
        for _, row in transfers_df.iterrows():
            from_cat = str(row["from_category"])
            moved = float(row["amount_moved"])
            prio = int(row["from_priority"])
            rem = float(remaining_map.get(from_cat, 0.0))

            st.markdown(
                f"- **Move €{moved:,.0f}** from **{from_cat}** → **{target_cat}**  \n"
                f"  *Why:* {from_cat} is priority **{prio}** and still has about **€{rem:,.0f}** remaining."
            )

        uncovered = float(transfers_df.attrs.get("uncovered_amount", 0.0))
        if uncovered > 0:
            st.warning(f"Safety caps prevented covering the full overspend. Remaining uncovered: **€{uncovered:,.0f}**.")

        with st.expander("See details (table)"):
            st.dataframe(transfers_df, use_container_width=True)

        apply = st.checkbox("Apply reallocation (prototype action)")
        if apply:
            new_df = budgets_df.copy()

            # reduce source budgets
            for _, t in transfers_df.iterrows():
                new_df.loc[new_df["category"] == t["from_category"], "budget"] -= t["amount_moved"]

            # increase target budget
            moved_total = float(transfers_df["amount_moved"].sum())
            new_df.loc[new_df["category"] == target_cat, "budget"] += moved_total

            new_df["remaining"] = new_df["budget"] - new_df["spent_so_far"]

            st.success("Reallocation applied (prototype).")
            st.dataframe(new_df[["category", "budget", "priority", "spent_so_far", "remaining"]], use_container_width=True)

st.divider()


# -----------------------------
# 5) How AI would improve in real product
# -----------------------------
with st.expander("How AI would improve this in a real product (optional explanation)"):
    st.markdown(
        """
- **Transaction classification**: Automatically assign each transaction to a category.
- **Personalized forecasting**: Use your own historical data (seasonality + habits), not synthetic history.
- **Smarter recommendations**: Optimize reallocations with constraints (protected categories, minimum budgets).
- **Preference learning**: Learn which suggestions you accept and adapt future guidance.
        """
    )
