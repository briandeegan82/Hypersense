#!/usr/bin/env python3
"""
ARW File Viewer
A Python script to open and display Sony .ARW (RAW) image files.

Requirements:
- rawpy: pip install rawpy
- matplotlib: pip install matplotlib
- numpy: pip install numpy

Usage:
    python display_arw.py [path_to_arw_file]
    or
    python display_arw.py  # will look for ARW files in input_images/ directory
"""

import os
import sys
import glob
import rawpy
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def find_arw_files(directory="input_images"):
    """Find all ARW files in the specified directory."""
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found.")
        return []
    
    arw_files = glob.glob(os.path.join(directory, "*.ARW")) + glob.glob(os.path.join(directory, "*.arw"))
    return arw_files


def display_arw_file(file_path):
    """Display an ARW file using matplotlib."""
    try:
        print(f"Loading ARW file: {file_path}")
        
        # Read the RAW file
        with rawpy.imread(file_path) as raw:
            # Get basic info about the raw file
            try:
                camera_model = getattr(raw, 'camera_model', 'Unknown')
                print(f"Camera: {camera_model}")
            except:
                print("Camera: Unknown")
            
            print(f"Image size: {raw.sizes.width} x {raw.sizes.height}")
            
            try:
                color_desc = getattr(raw, 'color_description', 'Unknown')
                print(f"Color description: {color_desc}")
            except:
                print("Color description: Unknown")
            
            # Post-process the raw image to obtain an RGB image
            # You can adjust these parameters for different looks
            rgb_image = raw.postprocess(
                use_camera_wb=True,  # Use camera white balance
                half_size=False,     # Full resolution
                no_auto_bright=True, # Don't auto-adjust brightness
                output_bps=8         # 8-bit output
            )
            
            print(f"Processed image shape: {rgb_image.shape}")
            print(f"Data type: {rgb_image.dtype}")
            print(f"Value range: {rgb_image.min()} - {rgb_image.max()}")
            
            # Display the image
            plt.figure(figsize=(12, 8))
            plt.imshow(rgb_image)
            plt.title(f"ARW Image: {os.path.basename(file_path)}")
            plt.axis('off')
            
            # Add some basic image info as text
            try:
                camera_model = getattr(raw, 'camera_model', 'Unknown')
            except:
                camera_model = 'Unknown'
            info_text = f"Size: {rgb_image.shape[1]}x{rgb_image.shape[0]}\nCamera: {camera_model}"
            plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
            return True
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False


def main():
    """Main function to handle command line arguments and display ARW files."""
    if len(sys.argv) > 1:
        # If a specific file path is provided
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        if not file_path.lower().endswith(('.arw', '.ARW')):
            print("Please provide an ARW file.")
            return
            
        display_arw_file(file_path)
    else:
        # Look for ARW files in the input_images directory
        arw_files = find_arw_files()
        
        if not arw_files:
            print("No ARW files found in 'input_images/' directory.")
            print("Usage: python display_arw.py [path_to_arw_file]")
            return
        
        print(f"Found {len(arw_files)} ARW file(s):")
        for i, file_path in enumerate(arw_files, 1):
            print(f"{i}. {file_path}")
        
        if len(arw_files) == 1:
            # If only one file, display it directly
            display_arw_file(arw_files[0])
        else:
            # If multiple files, let user choose
            try:
                choice = input(f"\nEnter the number of the file to display (1-{len(arw_files)}): ")
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(arw_files):
                    display_arw_file(arw_files[choice_idx])
                else:
                    print("Invalid choice.")
            except (ValueError, KeyboardInterrupt):
                print("Operation cancelled.")


if __name__ == "__main__":
    main()
