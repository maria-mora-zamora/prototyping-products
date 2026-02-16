import streamlit as st
import pandas as pd

st.set_page_config(page_title="Priority-Aware Budget Assistant", layout="centered")

st.title("Priority-Aware Budget Assistant")
st.caption("Prototype of an AI-assisted dynamic budgeting feature for a banking app.")

BUFFER_NAME = "Overspend buffer (to stay within total budget)"

# -----------------------------
# 1) Budget setup (inputs)
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
        budget = st.number_input(
            "Budget (€)", min_value=0.0, value=float(default_budget), step=10.0, key=f"budget_{i}"
        )
    with c3:
        priority = st.slider("Priority", 1, 5, int(default_priority), key=f"prio_{i}")

    categories.append({"category": name.strip(), "budget": float(budget), "priority": int(priority)})

# Remove empty names
categories = [c for c in categories if c["category"] != ""]

if len(categories) == 0:
    st.error("Please add at least one category (category name cannot be empty).")
    st.stop()

budgets_df = pd.DataFrame(categories)

# Allocation check (informative)
total_planned = budgets_df["budget"].sum()
allocation_gap = total_budget - total_planned  # + = unallocated, - = overallocated

st.subheader("Allocation check")

cA, cB = st.columns(2)
cA.metric("Total allocated across categories (€)", f"{total_planned:,.0f}")
cB.metric("Remaining to allocate (€)", f"{allocation_gap:,.0f}")

if allocation_gap > 0:
    st.info(
        f"You still have **€{allocation_gap:,.0f}** unallocated. "
        "You can distribute it across categories."
    )
elif allocation_gap < 0:
    st.error(
        f"You are **€{abs(allocation_gap):,.0f}** over budget. "
        "Reduce one or more category budgets to stay within the total limit."
    )
else:
    st.success("Perfect! Your category budgets add up exactly to your total monthly budget.")

if total_budget > 0:
    pct = min(total_planned / total_budget, 1.0)
    st.progress(pct, text=f"Allocated {total_planned:,.0f} / {total_budget:,.0f} (€)")

st.divider()

# -----------------------------
# 2) Spending monitoring (simulated for prototype)
# -----------------------------
st.header("2) Spending monitoring (prototype simulation)")

st.write(
    "For prototyping purposes, you can simulate current spending per category. "
    "Later, this could be replaced by real transaction data."
)

spending = []
for i, row in budgets_df.iterrows():
    spent = st.number_input(
        f"Current spent in {row['category']} (€)",
        min_value=0.0,
        value=max(0.0, row["budget"] * 0.6),
        step=10.0,
        key=f"spent_{i}",
    )
    spending.append(float(spent))

budgets_df["spent_so_far"] = spending
budgets_df["remaining"] = budgets_df["budget"] - budgets_df["spent_so_far"]
budgets_df["overspend_now"] = budgets_df["spent_so_far"] - budgets_df["budget"]  # positive means already overspent

total_spent = budgets_df["spent_so_far"].sum()

c1, c2, c3 = st.columns(3)
c1.metric("Planned category budgets (€)", f"{total_planned:,.0f}")
c2.metric("Total spent so far (€)", f"{total_spent:,.0f}")
c3.metric("Total budget limit (€)", f"{total_budget:,.0f}")

st.divider()

# -----------------------------
# 3) Risk detection (simple logic)
# -----------------------------
st.header("3) Risk detection")

day = st.slider("Day of month", 1, 31, 15)
days_in_month = 30  # simplification for prototype

pace_factor = days_in_month / day
projected_total = total_spent * pace_factor

st.write(f"**Projected end-of-month spending:** €{projected_total:,.0f}")

over_by = projected_total - total_budget

if over_by > 0:
    st.warning(f"At this pace, you may exceed your total budget by **€{over_by:,.0f}**.")
else:
    st.success("You are currently on track to stay within your total monthly budget.")

st.divider()

# -----------------------------
# 4) Recommendation: REALLOCATE budgets based on behavior + priorities
# -----------------------------
st.header("4) Recommended budget adjustment")

st.write(
    "If there is a risk of exceeding the total budget, the assistant suggests reallocating "
    "budget from lower-priority categories to protect higher-priority ones."
)

def suggest_reallocation_transfers(df: pd.DataFrame, amount_to_cover: float) -> pd.DataFrame:
    """
    Suggest reallocations (transfers) from categories into an 'Overspend buffer' to cover projected overspending.
    Rules:
    - Transfer from lowest-priority categories first
    - Only transfer from categories with remaining budget
    - Avoid extreme adjustments (cap at 30% of category budget)
    - Never reduce below spent_so_far (ensured by using 'remaining')
    """
    if amount_to_cover <= 0:
        return pd.DataFrame()

    candidates = df[df["remaining"] > 0].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates = candidates.sort_values(
        by=["priority", "remaining", "budget"],
        ascending=[True, False, False],
    )

    transfers = []
    remaining_needed = amount_to_cover

    for _, r in candidates.iterrows():
        if remaining_needed <= 0:
            break

        max_reducible = 0.30 * r["budget"]
        safe_reducible = min(max_reducible, r["remaining"])
        move_amount = min(safe_reducible, remaining_needed)

        if move_amount > 0:
            transfers.append(
                {
                    "from_category": r["category"],
                    "from_priority": int(r["priority"]),
                    "to": BUFFER_NAME,
                    "amount_moved": float(move_amount),
                }
            )
            remaining_needed -= move_amount

    transfers_df = pd.DataFrame(transfers)
    # Store how much we *couldn't* cover (if caps prevent reaching the target)
    if not transfers_df.empty:
        transfers_df.attrs["uncovered_amount"] = float(max(0.0, remaining_needed))
    return transfers_df


if over_by <= 0:
    st.info("No reallocation needed based on current spending pace.")
else:
    st.write(f"To stay within your total budget, you may need to reallocate about **€{over_by:,.0f}**.")

    transfers_df = suggest_reallocation_transfers(budgets_df, over_by)

    if transfers_df.empty:
        st.error(
            "No safe reallocation could be generated. "
            "This may happen if all categories are already fully spent (no remaining budget to reallocate)."
        )
    else:
        # --- Build a clearer explanation (what is the buffer + what's driving the issue?) ---
        overspenders = budgets_df[budgets_df["overspend_now"] > 0].copy()

        st.subheader("What’s happening?")

        if not overspenders.empty:
            main_driver = overspenders.sort_values("overspend_now", ascending=False).iloc[0]
            driver_cat = str(main_driver["category"])
            driver_amount = float(main_driver["overspend_now"])

            st.markdown(
                f"**{driver_cat} is already above its planned budget by about €{driver_amount:,.0f}.**  \n"
                f"To keep your *overall* monthly budget under control, we create an **{BUFFER_NAME}**. "
                f"This buffer represents the extra money you need to cover overspending without touching your high-priority plan."
            )
        else:
            st.markdown(
                f"**Your risk comes from your overall spending pace**, even if no single category is above its planned budget yet.  \n"
                f"We create an **{BUFFER_NAME}** to set aside the amount needed to stay within your total budget."
            )

        # Explanation list of transfers (more intuitive than a raw table)
        st.subheader("Suggested adjustments (reallocations)")

        remaining_map = budgets_df.set_index("category")["remaining"].to_dict()
        budget_map = budgets_df.set_index("category")["budget"].to_dict()
        spent_map = budgets_df.set_index("category")["spent_so_far"].to_dict()

        for _, row in transfers_df.iterrows():
            from_cat = str(row["from_category"])
            moved = float(row["amount_moved"])
            prio = int(row["from_priority"])

            planned_budget = float(budget_map.get(from_cat, 0.0))
            spent_so_far = float(spent_map.get(from_cat, 0.0))
            remaining_here = float(remaining_map.get(from_cat, 0.0))

            st.markdown(
                f"- **Move €{moved:,.0f}** from **{from_cat}** → **{BUFFER_NAME}**  \n"
                f"  **Why:** {from_cat} has priority **{prio}** and still has **€{remaining_here:,.0f}** remaining "
                f"(planned €{planned_budget:,.0f}, spent €{spent_so_far:,.0f})."
            )

        uncovered = float(transfers_df.attrs.get("uncovered_amount", 0.0))
        if uncovered > 0:
            st.warning(
                f"With the current safety limits (e.g., not cutting too aggressively), the prototype could only "
                f"cover part of the needed amount. Remaining uncovered: **€{uncovered:,.0f}**."
            )

        # Keep technical table available but not front-and-center
        with st.expander("See details (table)"):
            st.dataframe(transfers_df, use_container_width=True)

        # Apply button
        st.subheader("Apply suggestion?")
        apply = st.checkbox("Apply reallocation (prototype action)")

        if apply:
            new_df = budgets_df.copy()

            # Apply transfers: reduce budgets in source categories
            for _, t in transfers_df.iterrows():
                new_df.loc[new_df["category"] == t["from_category"], "budget"] -= t["amount_moved"]

            # Create buffer row with total moved
            buffer_amount = float(transfers_df["amount_moved"].sum())
            buffer_row = pd.DataFrame([{
                "category": BUFFER_NAME,
                "budget": buffer_amount,
                "priority": 5,
                "spent_so_far": 0.0,
                "remaining": buffer_amount,
                "overspend_now": 0.0
            }])

            # Recompute remaining after budget changes (spent stays the same)
            new_df["remaining"] = new_df["budget"] - new_df["spent_so_far"]
            new_df["overspend_now"] = new_df["spent_so_far"] - new_df["budget"]

            new_df = pd.concat([new_df, buffer_row], ignore_index=True)

            st.success("Reallocation applied (prototype).")
            st.write(
                "✅ The plan was adjusted by moving funds from lower-priority categories into the buffer, "
                "so you can absorb overspending while protecting higher-priority needs."
            )

            st.subheader("Updated plan (after reallocation)")
            st.dataframe(new_df[["category", "priority", "budget", "spent_so_far", "remaining"]], use_container_width=True)

st.divider()

# -----------------------------
# 5) What would be AI in a real version?
# -----------------------------
with st.expander("How AI would improve this in a real product (optional explanation)"):
    st.markdown(
        """
- **Transaction classification**: Automatically assign each expense to a category.
- **Personalized forecasting**: Predict end-of-month spending using your historical patterns (seasonality, habits).
- **Smarter recommendations**: Suggest reallocations that fit your behavior (not only simple rules).
- **User preferences learning**: Learn which suggestions you tend to accept and adapt future recommendations.
        """
    )
