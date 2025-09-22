#!/bin/bash
# Script to run the CFA pattern simulator

# Activate virtual environment
source venv/bin/activate

# Run the CFA simulator
python simulate_cfa_patterns.py "$@"
