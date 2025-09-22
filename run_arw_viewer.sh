#!/bin/bash
# Activation script for ARW viewer

echo "Activating virtual environment and running ARW viewer..."
source venv/bin/activate
python display_arw.py "$@"

