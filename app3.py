import streamlit as st
import pandas as pd

st.set_page_config(page_title="Priority-Aware Budget Assistant", layout="centered")

st.title("Priority-Aware Budget Assistant")
st.caption("Prototype of an AI-assisted dynamic budgeting feature for a banking app.")

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
budgets_df["overspend_now"] = budgets_df["spent_so_far"] - budgets_df["budget"]  # > 0 means already overspent

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
# 4) Recommendation: move budget from low-priority categories -> overspent category (Rule A)
# -----------------------------
st.header("4) Recommended budget reallocation")

st.write(
    "Rule A: If you have already exceeded the planned budget in a category, the assistant reallocates "
    "budget from lower-priority categories to cover that overspend, while keeping the overall plan coherent."
)

def suggest_transfers_to_target(df: pd.DataFrame, target_category: str, amount_needed: float) -> pd.DataFrame:
    """
    Suggest transfers FROM other categories TO the target category.
    - Transfer from lowest priority first
    - Only transfer from categories with remaining budget (so the suggestion is feasible)
    - Avoid extreme changes (cap at 30% of source budget)
    - Never reduce source below what has already been spent (ensured by using 'remaining')
    """
    if amount_needed <= 0:
        return pd.DataFrame()

    candidates = df[(df["category"] != target_category) & (df["remaining"] > 0)].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates = candidates.sort_values(
        by=["priority", "remaining", "budget"],
        ascending=[True, False, False],
    )

    transfers = []
    remaining_needed = amount_needed

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
                    "to_category": target_category,
                    "amount_moved": float(move_amount),
                }
            )
            remaining_needed -= move_amount

    transfers_df = pd.DataFrame(transfers)
    if not transfers_df.empty:
        transfers_df.attrs["uncovered_amount"] = float(max(0.0, remaining_needed))
    return transfers_df


overspenders = budgets_df[budgets_df["overspend_now"] > 0].copy()

if overspenders.empty:
    st.info(
        "No category is currently above its planned budget, so there is no direct reallocation to propose yet.\n\n"
        "Tip: Try increasing the 'Current spent' of a category above its planned budget to see reallocations."
    )
else:
    # Rule A: choose the category with the highest current overspend
    target_row = overspenders.sort_values("overspend_now", ascending=False).iloc[0]
    target_cat = str(target_row["category"])
    target_overspend = float(target_row["overspend_now"])

    st.subheader("What’s happening?")
    st.markdown(
        f"**{target_cat} is currently over budget.**  \n"
        f"- Planned for {target_cat}: **€{float(target_row['budget']):,.0f}**  \n"
        f"- Spent so far: **€{float(target_row['spent_so_far']):,.0f}**  \n"
        f"➡️ That’s **€{target_overspend:,.0f}** above plan."
    )

    st.markdown(
        "To keep your overall monthly plan under control, the assistant suggests **moving budget from "
        "lower-priority categories** (where you still have remaining budget) **into the overspent category**."
    )

    transfers_df = suggest_transfers_to_target(budgets_df, target_cat, target_overspend)

    if transfers_df.empty:
        st.error(
            "No safe reallocation could be generated. "
            "This may happen if there is no remaining budget available in other categories."
        )
    else:
        st.subheader("Suggested adjustments (easy-to-read)")

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
                f"- **Move €{moved:,.0f}** from **{from_cat}** → **{target_cat}**  \n"
                f"  **Why:** {from_cat} has priority **{prio}** and still has **€{remaining_here:,.0f}** remaining "
                f"(planned €{planned_budget:,.0f}, spent €{spent_so_far:,.0f})."
            )

        uncovered = float(transfers_df.attrs.get("uncovered_amount", 0.0))
        if uncovered > 0:
            st.warning(
                f"With the current safety limits (e.g., not cutting too aggressively), the prototype could not fully "
                f"cover the overspend. Remaining uncovered: **€{uncovered:,.0f}**."
            )

        with st.expander("See details (table)"):
            st.dataframe(transfers_df, use_container_width=True)

        st.subheader("Apply suggestion?")
        apply = st.checkbox("Apply reallocation (prototype action)")

        if apply:
            new_df = budgets_df.copy()

            # Reduce source budgets
            for _, t in transfers_df.iterrows():
                new_df.loc[new_df["category"] == t["from_category"], "budget"] -= t["amount_moved"]

            # Increase target budget by the moved total
            moved_total = float(transfers_df["amount_moved"].sum())
            new_df.loc[new_df["category"] == target_cat, "budget"] += moved_total

            # Recompute derived columns (spent stays the same)
            new_df["remaining"] = new_df["budget"] - new_df["spent_so_far"]
            new_df["overspend_now"] = new_df["spent_so_far"] - new_df["budget"]

            st.success("Reallocation applied (prototype).")
            st.write(
                "✅ The plan was adjusted by shifting budget from lower-priority categories into the overspent category.\n\n"
                "This is meant to help you stay in control without feeling punished—it's a suggestion, not a judgment."
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
