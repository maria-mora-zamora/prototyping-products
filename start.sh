#!/bin/bash
set -e

# Start Ollama in the background
ollama serve &

# Wait for Ollama to respond
sleep 5

# Download the model if it's not already downloaded
ollama pull llama3.2:3b

# Launch Streamlit
streamlit run app-LLM.py --server.port=8501 --server.address=0.0.0.0