#!/bin/bash
# One-command demo runner. From the project root, just run:  bash run_demo.sh
set -e

echo "Installing dependencies..."
pip install -q -r requirements.txt 2>/dev/null || pip install -q -r requirements.txt --break-system-packages

echo "Generating sample transaction data (swap in the real UCI file for final results — see README)..."
mkdir -p data/raw
python3 src/generate_sample_data.py

echo ""
echo "Running the full segmentation pipeline..."
cd src
python3 main.py
cd ..

echo ""
echo "Done. Outputs:"
echo "  - Figures:          outputs/figures/"
echo "  - Cluster data:     data/processed/cluster_assignments.csv"
echo "  - Interactive view: open dashboard/index.html in a browser"
