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
echo "  Alpha Command Center Launch Script"
echo "  Factor Operations Terminal v2.0"
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

echo " Launching Alpha Command Center..."
echo "   URL: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="
echo ""

# Launch Streamlit with custom theme
streamlit run src/dashboard_alpha_command_center.py \
    --server.port=$PORT \
    --server.address=localhost \
    --theme.base=dark \
    --theme.primaryColor="#00d4ff" \
    --theme.backgroundColor="#0a0a0a" \
    --theme.secondaryBackgroundColor="#1a1a2e" \
    --theme.textColor="#ffffff" \
    --theme.font="monospace"
