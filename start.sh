#!/bin/bash
set -e

echo "Starting LiveKit agent..."
uv run src/agent/agent.py start &

echo "Starting Streamlit app..."
exec streamlit run src/streamlit_app.py --server.port=8501 --server.address=0.0.0.0
