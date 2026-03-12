#!/bin/bash
set -e

ollama serve &
sleep 5

ollama pull tinyllama

streamlit run app-LLM.py --server.port=8501 --server.address=0.0.0.0