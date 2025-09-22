#!/bin/bash
# Script to run the pseudo-RGB image creator

# Activate virtual environment
source venv/bin/activate

# Run the pseudo-RGB creator
python create_pseudo_rgb.py "$@"
