import streamlit as st

st.title("Priority-Aware Budget Assistant")

st.write("Prototype of an AI-assisted dynamic budgeting feature.")

st.header("Budget Setup")

income = st.number_input("Monthly income", min_value=0)

st.subheader("Groceries")
groceries_budget = st.number_input("Groceries budget")
groceries_priority = st.slider("Groceries priority (1=low, 5=high)", 1, 5)

st.subheader("Eating Out")
eating_budget = st.number_input("Eating out budget")
eating_priority = st.slider("Eating out priority (1=low, 5=high)", 1, 5)

st.subheader("Leisure")
leisure_budget = st.number_input("Leisure budget")
leisure_priority = st.slider("Leisure priority (1=low, 5=high)", 1, 5)

