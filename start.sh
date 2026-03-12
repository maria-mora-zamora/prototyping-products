#!/bin/bash
set -e

ollama serve &
sleep 5

ollama pull phi3:mini

streamlit run app-LLM.py --server.port=${PORT:-8501} --server.address=0.0.0.0