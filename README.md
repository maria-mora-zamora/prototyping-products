# Assignment 2

# Priority-Aware Budget Assistant

Prototype of an AI-assisted dynamic budgeting feature for a banking app.

This project extends the original budgeting prototype with a **local LLM-based what-if planner**. Users can set a monthly budget, inspect current spending from real transaction history, forecast end-of-month spending, view reallocation suggestions, and then simulate a natural-language spending scenario such as:

> Reduce Eating out by 40% and Shopping by 30%. Keep Savings unchanged.

The app uses a **local LLM via Ollama** to convert that text into structured category-level adjustments, and then uses Python to recalculate the forecast and the reallocation plan.

---

## Main files

- `app-LLM.py` → main Streamlit app for Assignment 2
- `model_applied.py` → previous non-LLM version used as the base prototype
- `data/transactions_user_24_ccnum_567868110212.csv` → filtered dataset for one user
- `app.py`, `app2.py`, `app3.py`, `app4-simulation.py`, `app5-AI.py` → earlier iterations kept for traceability

---

## What the app does

The app has five sections:

1. **Budget setup**  
   Users define a total monthly budget, category budgets, and category priorities.

2. **Current spending**  
   The app aggregates real transaction data up to a selected day of the month.

3. **Data-driven forecast**  
   The forecast estimates end-of-month spending using historical cumulative spending curves by category.

4. **Reallocation recommendation**  
   If a category is forecasted to overspend, the app suggests moving budget from lower-priority categories with remaining budget.

5. **AI What-If Planner**  
   A local LLM interprets a natural-language scenario and outputs structured percentage adjustments by category. Python then recalculates the forecast and compares the result before vs. after the scenario.

---

## Why the LLM feature is not trivial

The LLM is not used as a simple chat interface. Instead, it is part of a multi-step pipeline:

- the user writes a scenario in natural language
- the local LLM converts that scenario into structured JSON adjustments
- Python validates the categories and values
- Python recalculates the forecast and reallocation plan using the LLM output
- the UI displays the impact of the scenario

This means the LLM output directly changes the system behaviour.

---

## Data

The original dataset is too large for GitHub, so the repository only includes a filtered subset for one user.

Expected path:

`data/transactions_user_24_ccnum_567868110212.csv`

If the file is renamed or moved, update the path in the sidebar input of the app.

---

## How to run

Install dependencies:

```bash
pip install -r requirements.txt
```

Install and start Ollama, then pull the model used by the app:

```bash
ollama pull llama3.2:3b
ollama run llama3.2:3b
```

In a second terminal, run the Streamlit app:

```bash
streamlit run app-LLM.py
```

---

## Example scenario

A good example to test the LLM feature is:

```text
Reduce Eating out by 40% and Shopping by 30%. Keep Savings unchanged.
```

This should generate category-level adjustments and produce a visible difference between the original forecast and the simulated forecast.

---

## Requirements

- Python 3.10+
- Streamlit
- pandas
- numpy
- requests
- Ollama installed locally

---

## Notes

- The prototype uses a **local LLM**, so it does not require a paid API.
- The forecasting logic and reallocation logic are deterministic Python components.
- The LLM is only used for interpreting natural-language scenarios into structured inputs for the simulation.

---

# Priority-Aware Budget Assistant (Prototype)

Prototype of an AI-assisted dynamic budgeting feature for a banking app.

**Core idea:** users set a total monthly budget + priorities per spending category, and the assistant uses **real transaction history** to forecast end-of-month spending and suggest **budget reallocations** from lower-priority categories to protect higher-priority ones.

---

## Repo structure

- `model_applied.py` → ✅ Main Streamlit app (final version)
- `data/` → small, reproducible dataset used by the app (1 user subset)
- `export_user_subset.ipynb` → notebook used to generate the 1-user subset from the original Kaggle dataset (not included in repo)
- `data_preparation.ipynb` → exploratory analysis / data prep notes
- `app.py`, `app2.py`, `app3.py`, `app4-simulation.py`, `app5-AI.py` → earlier iterations (kept for traceability)

---

## Data

The original dataset is too large for GitHub, so this repository includes only a filtered subset for one user, stored in data/.

Expected file location:
	•	data/transactions_user_24_ccnum_567868110212.csv

If you rename the file, update the path inside the Streamlit sidebar input.

---

## Run the app

From the repository root:

streamlit run model_applied.py

---

## Notes on reproducibility
	•	The current month and forecast day can be changed in the sidebar.
	•	The total budget is dynamic:
	•	If Auto-allocate is ON, category budgets rescale automatically when the total budget changes.
	•	If Auto-allocate is OFF, category budgets can be edited manually.
	•	The forecasting method is based on a data-driven cumulative spending curve learned from historical months (statistical AI approach).

---

## Requirements

- Python 3.10+ (tested with Python 3.12)
- Packages:
  - streamlit
  - pandas
  - numpy

Install dependencies:

```bash
pip install streamlit pandas numpy