#!/bin/bash
set -e

streamlit run app-LLM.py --server.port=${PORT:-8501} --server.address=0.0.0.0