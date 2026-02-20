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

## Requirements

- Python 3.10+ (tested with Python 3.12)
- Packages:
  - streamlit
  - pandas
  - numpy

Install dependencies:

```bash
pip install streamlit pandas numpy

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

Notes on reproducibility
	•	The current month and forecast day can be changed in the sidebar.
	•	The total budget is dynamic:
	•	If Auto-allocate is ON, category budgets rescale automatically when the total budget changes.
	•	If Auto-allocate is OFF, category budgets can be edited manually.
	•	The forecasting method is based on a data-driven cumulative spending curve learned from historical months (statistical AI approach).