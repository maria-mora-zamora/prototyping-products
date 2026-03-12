import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import re

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Priority-Aware Budget Assistant", layout="centered")
st.title("Priority-Aware Budget Assistant")
st.caption("Data-driven budgeting prototype using real transaction history (1 user subset).")

# -----------------------------
# Local LLM config (Ollama)
# -----------------------------
OLLAMA_MODEL = "tinyllama"
OLLAMA_URL = "http://localhost:11434/api/generate"

# -----------------------------
# Constants & mapping
# -----------------------------
DAYS_IN_MONTH = 30  # simplification for prototype

CATEGORY_MAP = {
    "grocery_pos": "Groceries",
    "grocery_net": "Groceries",
    "food_dining": "Eating out",
    "entertainment": "Leisure",
    "travel": "Leisure",
    "shopping_pos": "Shopping",
    "shopping_net": "Shopping",
    "gas_transport": "Transport",
    "home": "Home",
    "personal_care": "Personal care",
    "health_fitness": "Health",
    "kids_pets": "Kids & Pets",
    "misc_pos": "Misc",
    "misc_net": "Misc",
}

# -----------------------------
# Helpers
# -----------------------------
@st.cache_data
def load_user_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["year"] = df["trans_date_trans_time"].dt.year
    df["month"] = df["trans_date_trans_time"].dt.month
    df["day"] = df["trans_date_trans_time"].dt.day
    df["budget_category"] = df["category"].map(CATEGORY_MAP).fillna("Other")
    return df


def build_avg_cumulative_curve(df_train: pd.DataFrame) -> pd.DataFrame:
    """
    Build avg cumulative spend fraction per day per budget_category.
    Output: budget_category, day, cum_frac
    """
    if df_train.empty:
        return pd.DataFrame(columns=["budget_category", "day", "cum_frac"])

    daily_train = (
        df_train.groupby(["year", "month", "budget_category", "day"])["amt"]
        .sum()
        .reset_index()
    )

    monthly_totals = (
        daily_train.groupby(["year", "month", "budget_category"])["amt"]
        .sum()
        .reset_index()
        .rename(columns={"amt": "month_total"})
    )

    daily_train = daily_train.merge(
        monthly_totals,
        on=["year", "month", "budget_category"],
        how="left",
    )

    daily_train = daily_train.sort_values(["year", "month", "budget_category", "day"])
    daily_train["cum_spend"] = daily_train.groupby(["year", "month", "budget_category"])["amt"].cumsum()
    daily_train["cum_frac"] = daily_train["cum_spend"] / daily_train["month_total"]

    avg_curve = (
        daily_train.groupby(["budget_category", "day"])["cum_frac"]
        .mean()
        .reset_index()
    )

    avg_curve["cum_frac"] = avg_curve["cum_frac"].clip(0.01, 0.99)
    return avg_curve


def get_curve_frac(avg_curve: pd.DataFrame, category: str, day: int):
    """
    Return learned cum_frac for (category, day).
    If exact day isn't available, use nearest earlier day; else nearest later day.
    """
    c = avg_curve[avg_curve["budget_category"] == category]
    if c.empty:
        return None

    exact = c[c["day"] == day]
    if not exact.empty:
        return float(exact["cum_frac"].iloc[0])

    earlier = c[c["day"] <= day].sort_values("day", ascending=False)
    if not earlier.empty:
        return float(earlier["cum_frac"].iloc[0])

    later = c[c["day"] > day].sort_values("day", ascending=True)
    if not later.empty:
        return float(later["cum_frac"].iloc[0])

    return None


def forecast_end_of_month(spent_so_far: float, frac, day: int):
    """
    Forecast using learned curve if available, otherwise fallback to simple pace.
    Returns forecast and method label.
    """
    if frac is None:
        return spent_so_far * (DAYS_IN_MONTH / max(day, 1)), "pace_fallback"
    return spent_so_far / frac, "curve"


def highlight_gap(val):
    try:
        v = float(val)
    except Exception:
        return ""
    if v > 0:
        return "color: red; font-weight: 700;"
    return "color: green;"


def suggest_total_budget(df_current: pd.DataFrame, df_train: pd.DataFrame) -> float:
    """
    Choose a sensible default monthly budget from data.
    For demo, we set it slightly BELOW baseline to trigger recommendations.
    """
    cur_total = float(df_current["amt"].sum()) if not df_current.empty else np.nan

    if df_train.empty:
        train_avg = np.nan
    else:
        monthly_train = df_train.groupby(["year", "month"])["amt"].sum()
        train_avg = float(monthly_train.mean()) if len(monthly_train) else np.nan

    base = cur_total if (not np.isnan(cur_total) and cur_total > 0) else train_avg
    if np.isnan(base) or base <= 0:
        base = 2500.0

    demo_budget = base * 0.90
    return float(round(demo_budget / 10) * 10)


def suggest_transfers_to_target(budgets_df: pd.DataFrame, target_category: str, amount_needed: float) -> pd.DataFrame:
    """
    Suggest transfers from low-priority categories with remaining budget to the target category.
    Cap reductions at 30% of donor budget, and never reduce below already-spent.
    """
    if amount_needed <= 0:
        return pd.DataFrame()

    df = budgets_df.copy()
    df["remaining"] = df["budget"] - df["spent_so_far"]
    donors = df[(df["category"] != target_category) & (df["remaining"] > 0)].copy()

    if donors.empty:
        return pd.DataFrame()

    donors = donors.sort_values(by=["priority", "remaining", "budget"], ascending=[True, False, False])

    transfers = []
    remaining_needed = amount_needed

    for _, r in donors.iterrows():
        if remaining_needed <= 0:
            break

        max_reducible = 0.30 * float(r["budget"])
        safe_reducible = min(max_reducible, float(r["remaining"]))
        moved = min(safe_reducible, remaining_needed)

        if moved > 0:
            transfers.append(
                {
                    "from_category": r["category"],
                    "from_priority": int(r["priority"]),
                    "to_category": target_category,
                    "amount_moved": float(moved),
                }
            )
            remaining_needed -= moved

    out = pd.DataFrame(transfers)
    out.attrs["uncovered_amount"] = float(max(0.0, remaining_needed))
    return out


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_json_from_response(raw_text: str):
    raw_text = raw_text.strip()
    try:
        return json.loads(raw_text)
    except Exception:
        pass

    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw_text[start:end + 1]
        return json.loads(candidate)

    raise ValueError("No valid JSON found in model response.")


def call_ollama_json(prompt: str):
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
            },
            timeout=120,
        )

        if response.status_code != 200:
            return False, f"Ollama error ({response.status_code}): {response.text}"

        raw_text = response.json().get("response", "").strip()
        if not raw_text:
            return False, "Ollama returned an empty response."

        try:
            parsed = extract_json_from_response(raw_text)
            return True, parsed
        except Exception:
            return False, f"The model returned invalid JSON:\n\n{raw_text}"

    except requests.exceptions.ConnectionError:
        return False, "No connection to Ollama. Make sure Ollama is installed, open, and running."
    except Exception as e:
        return False, f"Request failed: {str(e)}"


def build_whatif_prompt(user_scenario: str, categories: list[str]):
    category_list = json.dumps(categories, ensure_ascii=False)

    return f"""
You convert natural-language budgeting scenarios into structured category adjustments.

User scenario:
"{user_scenario}"

Available categories:
{category_list}

Rules:
- Return ONLY valid JSON.
- Do not use markdown.
- Only use categories from the provided list.
- Negative values mean reducing forecasted spending.
- Positive values mean increasing forecasted spending.
- Use decimal percentages.
- If the user says "protect" a category, do not change it unless they also explicitly say to increase it.
- If the user says "reduce X by 30%", output -0.30.
- If the user says "cut", interpret it as a reduction.
- Keep the output conservative and literal.

Return exactly this JSON structure:
{{
  "scenario_summary": "one short sentence",
  "adjustments": {{
    "Eating out": -0.40,
    "Shopping": -0.30
  }}
}}
"""


def apply_scenario_to_forecast(forecast_df: pd.DataFrame, adjustments: dict) -> pd.DataFrame:
    scenario_df = forecast_df.copy()

    def safe_adjustment(cat):
        val = adjustments.get(cat, 0.0)
        try:
            return float(val)
        except Exception:
            return 0.0

    scenario_df["adjustment"] = scenario_df["category"].apply(safe_adjustment)
    scenario_df["scenario_forecast_end_month"] = scenario_df["forecast_end_month"] * (1 + scenario_df["adjustment"])
    scenario_df["scenario_overspend_vs_budget"] = scenario_df["scenario_forecast_end_month"] - scenario_df["budget"]
    return scenario_df


# -----------------------------
# Sidebar: data + date selection
# -----------------------------
st.sidebar.header("Data")

data_path = st.sidebar.text_input(
    "Path to filtered user CSV",
    value="data/transactions_user_24_ccnum_567868110212.csv",
)

try:
    df_user = load_user_csv(data_path)
except Exception as e:
    st.error(f"Could not load the CSV file: {e}")
    st.stop()

st.sidebar.write("Rows:", df_user.shape[0])
st.sidebar.write("Unique users:", df_user["cc_num"].nunique())

available_months = (
    df_user[["year", "month"]]
    .drop_duplicates()
    .sort_values(["year", "month"])
    .values
    .tolist()
)

month_labels = [f"{y}-{m:02d}" for y, m in available_months]
selected_label = st.sidebar.selectbox("Current month (forecast month)", month_labels, index=len(month_labels) - 1)

CURRENT_YEAR = int(selected_label.split("-")[0])
CURRENT_MONTH = int(selected_label.split("-")[1])

day = st.sidebar.slider("Day of month for forecast", 1, 30, 15)

df_train = df_user[
    (df_user["year"] < CURRENT_YEAR)
    | ((df_user["year"] == CURRENT_YEAR) & (df_user["month"] < CURRENT_MONTH))
].copy()

df_current = df_user[
    (df_user["year"] == CURRENT_YEAR) & (df_user["month"] == CURRENT_MONTH)
].copy()

if df_current.empty:
    st.error("No transactions found for the selected month. Choose another month in the sidebar.")
    st.stop()

st.divider()

# -----------------------------
# 1) Budget setup
# -----------------------------
st.header("1) Budget setup")

cats_in_data = (
    df_user["budget_category"]
    .value_counts()
    .index
    .tolist()
)

if "total_budget" not in st.session_state:
    st.session_state["total_budget"] = suggest_total_budget(df_current, df_train)

auto_allocate = st.checkbox("Auto-allocate category budgets from total budget", value=True)
st.session_state["auto_allocate"] = auto_allocate

if "budget_shares" not in st.session_state:
    if not df_train.empty:
        train_cat_totals = df_train.groupby("budget_category")["amt"].sum()
        train_cat_totals = train_cat_totals.reindex(cats_in_data).fillna(0.0)
        total = float(train_cat_totals.sum())
        if total > 0:
            shares = (train_cat_totals / total).to_dict()
        else:
            shares = {c: 1.0 / len(cats_in_data) for c in cats_in_data}
    else:
        shares = {c: 1.0 / len(cats_in_data) for c in cats_in_data}

    st.session_state["budget_shares"] = shares


def rescale_category_budgets():
    if not st.session_state.get("auto_allocate", True):
        return

    total = float(st.session_state["total_budget"])
    shares = st.session_state["budget_shares"]

    for i, cat in enumerate(cats_in_data):
        key = f"b_{i}"
        st.session_state[key] = float(round((total * float(shares.get(cat, 0.0))) / 10) * 10)


st.number_input(
    "Total monthly spending budget (€)",
    min_value=0.0,
    step=50.0,
    key="total_budget",
    on_change=rescale_category_budgets,
)

if auto_allocate:
    rescale_category_budgets()
else:
    for i, _ in enumerate(cats_in_data):
        key = f"b_{i}"
        if key not in st.session_state:
            st.session_state[key] = 100.0

st.markdown("Set budgets and priorities for each category (1 = low priority, 5 = high priority).")

budgets = []
for i, cat in enumerate(cats_in_data):
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        st.write(cat)

    with c2:
        st.number_input(
            "Budget (€)",
            min_value=0.0,
            step=10.0,
            key=f"b_{i}",
            disabled=auto_allocate,
        )

    with c3:
        p_key = f"p_{i}"
        if p_key not in st.session_state:
            st.session_state[p_key] = 3
        st.slider("Priority", 1, 5, key=p_key)

    budgets.append(
        {"category": cat, "budget": float(st.session_state[f"b_{i}"]), "priority": int(st.session_state[p_key])}
    )

budgets_df = pd.DataFrame(budgets)

total_budget = float(st.session_state["total_budget"])
total_planned = float(budgets_df["budget"].sum())
allocation_gap = float(total_budget - total_planned)

st.subheader("Allocation check")
cA, cB = st.columns(2)
cA.metric("Total allocated (€)", f"{total_planned:,.0f}")
cB.metric("Remaining to allocate (€)", f"{allocation_gap:,.0f}")

if allocation_gap > 0:
    st.info(f"You still have **€{allocation_gap:,.0f}** unallocated.")
elif allocation_gap < 0:
    st.error(f"You are **€{abs(allocation_gap):,.0f}** over your total budget.")
else:
    st.success("Perfect: category budgets sum exactly to the total budget.")

st.divider()

# -----------------------------
# 2) Current spending
# -----------------------------
st.header("2) Current spending (from real transactions)")

spent_so_far = (
    df_current[df_current["day"] <= day]
    .groupby("budget_category")["amt"]
    .sum()
    .reset_index()
    .rename(columns={"budget_category": "category", "amt": "spent_so_far"})
)

budgets_df = budgets_df.merge(spent_so_far, on="category", how="left").fillna({"spent_so_far": 0.0})
budgets_df["remaining"] = budgets_df["budget"] - budgets_df["spent_so_far"]

st.dataframe(budgets_df[["category", "budget", "priority", "spent_so_far", "remaining"]], use_container_width=True)
st.metric("Total spent so far (€)", f"{float(budgets_df['spent_so_far'].sum()):,.0f}")

st.divider()

# -----------------------------
# 3) Data-driven Forecast
# -----------------------------
st.header("3) Data-driven Forecast (AI core)")

avg_curve = build_avg_cumulative_curve(df_train)

forecast_rows = []
for _, r in budgets_df.iterrows():
    cat = r["category"]
    spent = float(r["spent_so_far"])

    frac = get_curve_frac(avg_curve, cat, day)
    forecast, method = forecast_end_of_month(spent, frac, day)

    gap = float(forecast) - float(r["budget"])
    forecast_rows.append(
        {
            "category": cat,
            "budget": float(r["budget"]),
            "spent_so_far": spent,
            "forecast_end_month": float(forecast),
            "forecast_overspend_vs_budget": gap,
            "method": method,
        }
    )

forecast_df = pd.DataFrame(forecast_rows)
styled = forecast_df.style.map(highlight_gap, subset=["forecast_overspend_vs_budget"])
st.dataframe(styled, use_container_width=True)

forecast_total = float(forecast_df["forecast_end_month"].sum())
st.metric("Forecast total end-of-month spend (€)", f"{forecast_total:,.0f}")

if forecast_total > total_budget:
    st.warning(f"Forecast suggests you may exceed your TOTAL monthly budget by about €{forecast_total - total_budget:,.0f}.")
else:
    st.success("Forecast suggests you're on track to stay within your total monthly budget.")

with st.expander("Explainability (optional): learned curve sample"):
    st.dataframe(avg_curve.head(30), use_container_width=True)

st.divider()

# -----------------------------
# 4) Reallocation recommendation
# -----------------------------
st.header("4) Reallocation recommendation")

candidates = forecast_df[forecast_df["forecast_overspend_vs_budget"] > 0].copy()

target_cat = None
amount_needed = 0.0
transfers_df = pd.DataFrame()

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
        remaining_map = budgets_df.set_index("category")["remaining"].to_dict()

        st.subheader("Suggested adjustment (easy-to-read)")
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

        with st.expander("See transfers table"):
            st.dataframe(transfers_df, use_container_width=True)

st.divider()

# -----------------------------
# 5) AI What-If Planner
# -----------------------------
st.header("5) AI What-If Planner")
st.caption("Describe a spending-change scenario in natural language. The local LLM converts it into category-level percentage adjustments, and Python recalculates the forecast and the reallocation plan.")

default_scenario = "Reduce Eating out by 40% and Shopping by 30%. Keep Savings unchanged."

scenario_text = st.text_area(
    "Scenario",
    value=default_scenario,
    height=110,
    help="Example: Reduce Eating out by 40% and Shopping by 30%. Keep Savings unchanged."
)

if st.button("Simulate scenario"):
    prompt = build_whatif_prompt(scenario_text, cats_in_data)

    with st.spinner("Interpreting scenario with local LLM..."):
        ok, llm_result = call_ollama_json(prompt)

    if not ok:
        st.error(llm_result)
        st.info("Check that Ollama is running and that you already tested: ollama run llama3.2:3b")
    else:
        scenario_summary = clean_text(llm_result.get("scenario_summary", ""))
        adjustments = llm_result.get("adjustments", {}) or {}

        valid_adjustments = {}
        for cat, value in adjustments.items():
            if cat in cats_in_data:
                try:
                    valid_adjustments[cat] = float(value)
                except Exception:
                    pass

        st.subheader("AI interpretation")
        if scenario_summary:
            st.write(scenario_summary)
        st.json(valid_adjustments)

        scenario_forecast_df = apply_scenario_to_forecast(forecast_df, valid_adjustments)

        st.subheader("Forecast after scenario")
        st.dataframe(
            scenario_forecast_df[
                [
                    "category",
                    "budget",
                    "forecast_end_month",
                    "adjustment",
                    "scenario_forecast_end_month",
                    "scenario_overspend_vs_budget",
                ]
            ],
            use_container_width=True,
        )

        total_before = float(forecast_df["forecast_end_month"].sum())
        total_after = float(scenario_forecast_df["scenario_forecast_end_month"].sum())

        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast before (€)", f"{total_before:,.0f}")
        c2.metric("Forecast after (€)", f"{total_after:,.0f}")
        c3.metric("Estimated change (€)", f"{total_after - total_before:,.0f}")

        scenario_candidates = scenario_forecast_df[
            scenario_forecast_df["scenario_overspend_vs_budget"] > 0
        ].copy()

        st.subheader("Reallocation after scenario")

        if scenario_candidates.empty:
            st.success("Under this scenario, no category is forecasted to exceed its budget.")
        else:
            scenario_target = scenario_candidates.sort_values(
                "scenario_overspend_vs_budget", ascending=False
            ).iloc[0]

            scenario_target_cat = str(scenario_target["category"])
            scenario_amount_needed = float(scenario_target["scenario_overspend_vs_budget"])

            st.write(
                f"After applying the scenario, **{scenario_target_cat}** is forecasted to overspend by **€{scenario_amount_needed:,.0f}**."
            )

            scenario_transfers_df = suggest_transfers_to_target(
                budgets_df, scenario_target_cat, scenario_amount_needed
            )

            if scenario_transfers_df.empty:
                st.error("No safe reallocation available after applying the scenario.")
            else:
                remaining_map = budgets_df.set_index("category")["remaining"].to_dict()

                for _, row in scenario_transfers_df.iterrows():
                    from_cat = str(row["from_category"])
                    moved = float(row["amount_moved"])
                    prio = int(row["from_priority"])
                    rem = float(remaining_map.get(from_cat, 0.0))

                    st.markdown(
                        f"- **Move €{moved:,.0f}** from **{from_cat}** → **{scenario_target_cat}**  \n"
                        f"  *Why:* {from_cat} is priority **{prio}** and still has about **€{rem:,.0f}** remaining."
                    )

                uncovered = float(scenario_transfers_df.attrs.get("uncovered_amount", 0.0))
                if uncovered > 0:
                    st.warning(
                        f"Safety caps prevented covering the full overspend after the scenario. Remaining uncovered: **€{uncovered:,.0f}**."
                    )

        st.subheader("Before vs after scenario")
        compare_df = scenario_forecast_df[
            ["category", "forecast_end_month", "scenario_forecast_end_month"]
        ].copy()
        compare_df["difference"] = compare_df["scenario_forecast_end_month"] - compare_df["forecast_end_month"]
        st.dataframe(compare_df, use_container_width=True)