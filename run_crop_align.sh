#!/bin/bash
# Script to run the ARW cropping and alignment tool

# Activate virtual environment
source venv/bin/activate

# Run the cropping and alignment script
python crop_align_arw.py "$@"
