#!/bin/bash
set -e

echo "Downloading required model files..."
uv run src/agent/agent.py download-files

echo "Starting LiveKit agent..."
uv run src/agent/agent.py start &

echo "Waiting 30 seconds before starting Streamlit..."
sleep 30

echo "Starting Streamlit app..."
exec streamlit run src/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
