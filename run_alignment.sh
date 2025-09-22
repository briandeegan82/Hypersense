#!/bin/bash

# Multispectral Image Alignment Script Runner
# This script activates the virtual environment and runs the alignment script

# Activate virtual environment
source venv/bin/activate

# Check if an ARW file is provided as argument
if [ $# -eq 0 ]; then
    echo "Usage: $0 [ARW_file_path] [options]"
    echo ""
    echo "Examples:"
    echo "  $0 input_images/DSC00204.ARW"
    echo "  $0 input_images/DSC00204.ARW --layout 2x3 --save"
    echo "  $0 input_images/DSC00204.ARW --reference 2 --save"
    echo ""
    echo "Available options:"
    echo "  --layout {auto,2x3,3x2,1x6,6x1}  Band layout pattern (default: auto)"
    echo "  --reference {0-5}                 Reference band index (default: 0)"
    echo "  --save                           Save aligned bands as individual images"
    echo "  --no-display                     Skip displaying the visualization"
    echo ""
    echo "If no ARW file is provided, the script will look for ARW files in input_images/"
    echo ""
    # Run without arguments to show file selection
    python align_multispectral.py
else
    # Run with provided arguments
    python align_multispectral.py "$@"
fi

