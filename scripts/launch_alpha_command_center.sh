#!/bin/bash
# Launch the Alpha Command Center - Factor Operations Terminal
# 
# Usage: ./scripts/launch_alpha_command_center.sh [--port PORT]

set -e

# Default port
PORT=8502

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --port)
      PORT="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

echo "=========================================="
echo "  Equity Factors Dashboard"
echo "  (formerly: Alpha Command Center)"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo " Error: Must run from project root directory"
    exit 1
fi

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
    echo " Virtual environment activated"
else
    echo "  Warning: Virtual environment not found at .venv/"
fi

# Check if data exists
if [ ! -f "factor_returns.csv" ]; then
    echo "  Warning: factor_returns.csv not found"
    echo "   Run: python -m src discover --universe VTHR"
    echo ""
fi

echo " Launching dashboard..."
echo "   URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Launch Streamlit (theme configured in .streamlit/config.toml)
streamlit run src/dashboard.py \
    --server.port=$PORT \
    --server.address=localhost
