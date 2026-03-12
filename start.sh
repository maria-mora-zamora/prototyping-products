#!/bin/bash
set -e

ollama serve &
sleep 5

ollama pull llama3.2:3b

streamlit run app-LLM.py --server.port=${PORT:-8501} --server.address=0.0.0.0