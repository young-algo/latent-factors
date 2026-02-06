#!/bin/bash
# Launch the Equity Factors Dashboard (Streamlit)
#
# Usage:
#   ./scripts/launch_dashboard.sh [--port PORT]

set -e

PORT=8501

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
echo "=========================================="
echo ""

if [ ! -f "pyproject.toml" ]; then
  echo "Error: Must run from project root directory"
  exit 1
fi

if [ -f ".venv/bin/activate" ]; then
  source .venv/bin/activate
  echo "Virtual environment activated"
else
  echo "Warning: Virtual environment not found at .venv/"
fi

echo ""
echo "Launching dashboard..."
echo "URL: http://localhost:$PORT"
echo "Press Ctrl+C to stop"
echo ""

streamlit run src/dashboard.py \
  --server.port="$PORT" \
  --server.address=localhost

