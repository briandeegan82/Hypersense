#!/usr/bin/env python3
"""
CFA Pattern Simulator for Multi-Spectral Images
Simulates different Color Filter Array (CFA) patterns using weighted combinations 
of the 14 spectral bands from multi-spectral camera data.

Supported CFA patterns:
- Bayer (RGGB): Red, Green, Green, Blue
- RCCB: Red, Clear, Clear, Blue  
- RCCG: Red, Clear, Clear, Green
- RYYCy: Red, Yellow, Yellow, Cyan

Requirements:
- numpy: pip install numpy
- matplotlib: pip install matplotlib
- opencv-python: pip install opencv-python

Usage:
    python simulate_cfa_patterns.py [path_to_cropped_images_directory]
    or
    python simulate_cfa_patterns.py  # will look for cropped_images_DSC00204/ directory
"""

import os
import sys
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def load_spectral_bands(directory):
    """Load all individual spectral bands from the cropped images directory."""
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found.")
        return None
    
    # Define spectral band information for each position
    spectral_info = [
        {"name": "top_left", "bands": {"R": 850, "G": None, "B": None}},
        {"name": "top_middle", "bands": {"R": 650, "G": 525, "B": None}},
        {"name": "top_right", "bands": {"R": 710, "G": 570, "B": 405}},
        {"name": "bottom_left", "bands": {"R": 650, "G": 550, "B": 430}},
        {"name": "bottom_middle", "bands": {"R": 735, "G": None, "B": 490}},
        {"name": "bottom_right", "bands": {"R": 685, "G": 560, "B": 450}}
    ]
    
    all_bands = []
    
    # Extract base name from directory (e.g., "DSC00111" from "cropped_images_DSC00111")
    base_name = os.path.basename(directory).replace("cropped_images_", "")
    
    for info in spectral_info:
        # Load the image file
        image_path = os.path.join(directory, f"{base_name}_{info['name']}.png")
        if not os.path.exists(image_path):
            print(f"Warning: Image file not found: {image_path}")
            continue
            
        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        bands = info["bands"]
        
        # Extract R, G, B channels if they exist
        if bands["R"] is not None:
            all_bands.append({
                "wavelength": bands["R"],
                "channel": "R",
                "data": image[:, :, 0],  # Red channel
                "position": info["name"]
            })
        
        if bands["G"] is not None:
            all_bands.append({
                "wavelength": bands["G"],
                "channel": "G", 
                "data": image[:, :, 1],  # Green channel
                "position": info["name"]
            })
        
        if bands["B"] is not None:
            all_bands.append({
                "wavelength": bands["B"],
                "channel": "B",
                "data": image[:, :, 2],  # Blue channel
                "position": info["name"]
            })
    
    # Sort by wavelength
    all_bands.sort(key=lambda x: x["wavelength"])
    
    # Reassign indices after sorting
    for i, band in enumerate(all_bands):
        band["index"] = i
    
    return all_bands


def create_ir_cutoff_filters():
    """Define different IR cut-off filter response curves."""
    
    wavelengths = [405, 430, 450, 490, 525, 550, 560, 570, 650, 685, 710, 735, 850]
    
    # No IR cut-off (passes all wavelengths)
    no_filter = np.ones(len(wavelengths))
    
    # Standard IR cut-off at ~650nm (typical for consumer cameras)
    standard_ir = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.3, 0.1, 0.05, 0.01])
    
    # Sharp IR cut-off at ~700nm (professional cameras)
    sharp_ir = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.9, 0.2, 0.05, 0.01])
    
    # Extended IR cut-off at ~800nm (allows more NIR)
    extended_ir = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.8, 0.3])
    
    # No IR cut-off (full spectrum)
    full_spectrum = np.ones(len(wavelengths))
    
    return {
        "wavelengths": wavelengths,
        "no_filter": no_filter,
        "standard_ir": standard_ir,
        "sharp_ir": sharp_ir,
        "extended_ir": extended_ir,
        "full_spectrum": full_spectrum
    }


def create_spectral_response_curves(ir_filter_type="standard_ir"):
    """Define spectral response curves for different color channels with IR cut-off filter."""
    
    # Define wavelength ranges for different color responses
    wavelengths = [405, 430, 450, 490, 525, 550, 560, 570, 650, 685, 710, 735, 850]
    
    # Get IR cut-off filter response
    ir_filters = create_ir_cutoff_filters()
    ir_response = ir_filters[ir_filter_type]
    
    # Red channel response (peaks around 600-700nm)
    red_response = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 0.9, 0.95, 1.0, 0.9, 0.7, 0.5, 0.2])
    
    # Green channel response (peaks around 500-600nm)
    green_response = np.array([0.1, 0.3, 0.5, 0.8, 1.0, 0.9, 0.8, 0.7, 0.3, 0.2, 0.1, 0.1, 0.05])
    
    # Blue channel response (peaks around 400-500nm)
    blue_response = np.array([0.8, 1.0, 0.9, 0.7, 0.4, 0.2, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05])
    
    # Clear channel response (broad spectrum, peaks in visible)
    clear_response = np.array([0.6, 0.8, 0.9, 1.0, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1])
    
    # Yellow channel response (combination of red and green)
    yellow_response = (red_response + green_response) / 2
    
    # Cyan channel response (combination of green and blue)
    cyan_response = (green_response + blue_response) / 2
    
    # Apply IR cut-off filter to all responses
    red_response *= ir_response
    green_response *= ir_response
    blue_response *= ir_response
    clear_response *= ir_response
    yellow_response *= ir_response
    cyan_response *= ir_response
    
    return {
        "wavelengths": wavelengths,
        "ir_filter_type": ir_filter_type,
        "ir_response": ir_response,
        "red": red_response,
        "green": green_response,
        "blue": blue_response,
        "clear": clear_response,
        "yellow": yellow_response,
        "cyan": cyan_response
    }


def simulate_color_channel(all_bands, target_wavelengths, response_curve, channel_name):
    """Simulate a color channel using weighted combination of spectral bands."""
    
    # Get the response curve for the target wavelengths
    response_values = []
    for wavelength in target_wavelengths:
        # Find closest wavelength in response curve
        closest_idx = np.argmin(np.abs(np.array(response_curve["wavelengths"]) - wavelength))
        response_values.append(response_curve[channel_name][closest_idx])
    
    # Normalize response values
    response_values = np.array(response_values)
    if np.sum(response_values) > 0:
        response_values = response_values / np.sum(response_values)
    
    # Create weighted combination
    height, width = all_bands[0]["data"].shape
    combined_channel = np.zeros((height, width), dtype=np.float32)
    
    for i, band in enumerate(all_bands):
        if i < len(response_values):
            combined_channel += band["data"].astype(np.float32) * response_values[i]
    
    # Convert back to uint8
    combined_channel = np.clip(combined_channel, 0, 255).astype(np.uint8)
    
    return combined_channel


def simulate_bayer_cfa(all_bands, response_curves):
    """Simulate Bayer (RGGB) CFA pattern."""
    print("Simulating Bayer (RGGB) CFA...")
    
    # Get target wavelengths for each channel
    target_wavelengths = [band["wavelength"] for band in all_bands]
    
    # Simulate R, G, B channels
    red_channel = simulate_color_channel(all_bands, target_wavelengths, response_curves, "red")
    green_channel = simulate_color_channel(all_bands, target_wavelengths, response_curves, "green")
    blue_channel = simulate_color_channel(all_bands, target_wavelengths, response_curves, "blue")
    
    # Create RGB image
    bayer_rgb = np.stack([red_channel, green_channel, blue_channel], axis=2)
    
    return bayer_rgb, {"red": red_channel, "green": green_channel, "blue": blue_channel}


def simulate_rccb_cfa(all_bands, response_curves):
    """Simulate RCCB CFA pattern."""
    print("Simulating RCCB CFA...")
    
    target_wavelengths = [band["wavelength"] for band in all_bands]
    
    # Simulate R, Clear, B channels
    red_channel = simulate_color_channel(all_bands, target_wavelengths, response_curves, "red")
    clear_channel = simulate_color_channel(all_bands, target_wavelengths, response_curves, "clear")
    blue_channel = simulate_color_channel(all_bands, target_wavelengths, response_curves, "blue")
    
    # Create RGB image (using clear for green channel)
    rccb_rgb = np.stack([red_channel, clear_channel, blue_channel], axis=2)
    
    return rccb_rgb, {"red": red_channel, "clear": clear_channel, "blue": blue_channel}


def simulate_rccg_cfa(all_bands, response_curves):
    """Simulate RCCG CFA pattern."""
    print("Simulating RCCG CFA...")
    
    target_wavelengths = [band["wavelength"] for band in all_bands]
    
    # Simulate R, Clear, G channels
    red_channel = simulate_color_channel(all_bands, target_wavelengths, response_curves, "red")
    clear_channel = simulate_color_channel(all_bands, target_wavelengths, response_curves, "clear")
    green_channel = simulate_color_channel(all_bands, target_wavelengths, response_curves, "green")
    
    # Create RGB image (using clear for blue channel)
    rccg_rgb = np.stack([red_channel, clear_channel, green_channel], axis=2)
    
    return rccg_rgb, {"red": red_channel, "clear": clear_channel, "green": green_channel}


def simulate_ryycy_cfa(all_bands, response_curves):
    """Simulate RYYCy CFA pattern."""
    print("Simulating RYYCy CFA...")
    
    target_wavelengths = [band["wavelength"] for band in all_bands]
    
    # Simulate R, Yellow, Cyan channels
    red_channel = simulate_color_channel(all_bands, target_wavelengths, response_curves, "red")
    yellow_channel = simulate_color_channel(all_bands, target_wavelengths, response_curves, "yellow")
    cyan_channel = simulate_color_channel(all_bands, target_wavelengths, response_curves, "cyan")
    
    # Create RGB image (using yellow for green, cyan for blue)
    ryycy_rgb = np.stack([red_channel, yellow_channel, cyan_channel], axis=2)
    
    return ryycy_rgb, {"red": red_channel, "yellow": yellow_channel, "cyan": cyan_channel}


def create_cfa_comparison(cfa_results, output_path="cfa_comparison.png"):
    """Create a comparison grid of all CFA simulations."""
    
    # Use 2x2 grid for 4 CFA patterns
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    cfa_names = ["Bayer (RGGB)", "RCCB", "RCCG", "RYYCy"]
    
    for i, (name, (rgb_image, channels)) in enumerate(cfa_results.items()):
        if i < len(axes):
            axes[i].imshow(rgb_image)
            axes[i].set_title(f"{name} CFA", fontsize=14, pad=10)
            axes[i].axis('off')
    
    # Hide unused subplots if any
    for i in range(len(cfa_results), len(axes)):
        axes[i].axis('off')
    
    plt.suptitle("CFA Pattern Simulations - Multi-Spectral Data", fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"CFA comparison saved: {output_path}")


def save_cfa_images(cfa_results, output_dir="cfa_simulations"):
    """Save individual CFA simulation images."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    saved_files = []
    
    for name, (rgb_image, channels) in cfa_results.items():
        # Save RGB image
        filename = f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Convert RGB to BGR for OpenCV
        rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, rgb_bgr)
        saved_files.append(filepath)
        print(f"Saved: {filepath}")
        
        # Save individual channels
        for channel_name, channel_data in channels.items():
            channel_filename = f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{channel_name}.png"
            channel_filepath = os.path.join(output_dir, channel_filename)
            cv2.imwrite(channel_filepath, channel_data)
            print(f"Saved: {channel_filepath}")
    
    return saved_files


def display_spectral_response_curves(response_curves, output_path="spectral_response_curves.png"):
    """Display the spectral response curves used for CFA simulation."""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    wavelengths = response_curves["wavelengths"]
    ir_filter_type = response_curves.get("ir_filter_type", "standard_ir")
    
    # Plot IR cut-off filter response
    ax1.plot(wavelengths, response_curves["ir_response"], 'k-', linewidth=3, label=f'IR Cut-off Filter ({ir_filter_type})')
    ax1.fill_between(wavelengths, 0, response_curves["ir_response"], alpha=0.3, color='gray')
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Transmission', fontsize=12)
    ax1.set_title(f'IR Cut-off Filter Response - {ir_filter_type.replace("_", " ").title()}', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(400, 900)
    ax1.set_ylim(0, 1.1)
    
    # Plot color channel responses
    ax2.plot(wavelengths, response_curves["red"], 'r-', linewidth=2, label='Red')
    ax2.plot(wavelengths, response_curves["green"], 'g-', linewidth=2, label='Green')
    ax2.plot(wavelengths, response_curves["blue"], 'b-', linewidth=2, label='Blue')
    ax2.plot(wavelengths, response_curves["clear"], 'k-', linewidth=2, label='Clear')
    ax2.plot(wavelengths, response_curves["yellow"], 'y-', linewidth=2, label='Yellow')
    ax2.plot(wavelengths, response_curves["cyan"], 'c-', linewidth=2, label='Cyan')
    
    ax2.set_xlabel('Wavelength (nm)', fontsize=12)
    ax2.set_ylabel('Relative Response', fontsize=12)
    ax2.set_title('Color Channel Response Curves (with IR Cut-off Applied)', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(400, 900)
    ax2.set_ylim(0, 1.1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Spectral response curves saved: {output_path}")


def main():
    """Main function to simulate CFA patterns."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simulate CFA patterns with configurable IR cut-off filters')
    parser.add_argument('directory', nargs='?', default='cropped_images_DSC00204',
                       help='Directory containing cropped spectral images (default: cropped_images_DSC00204)')
    parser.add_argument('--ir-filter', choices=['no_filter', 'standard_ir', 'sharp_ir', 'extended_ir', 'full_spectrum'],
                       default='standard_ir', help='IR cut-off filter type (default: standard_ir)')
    parser.add_argument('--list-filters', action='store_true',
                       help='List available IR filter types and exit')
    
    args = parser.parse_args()
    
    if args.list_filters:
        print("Available IR cut-off filter types:")
        print("  no_filter      - No IR cut-off (passes all wavelengths)")
        print("  standard_ir    - Standard IR cut-off at ~650nm (typical consumer cameras)")
        print("  sharp_ir       - Sharp IR cut-off at ~700nm (professional cameras)")
        print("  extended_ir    - Extended IR cut-off at ~800nm (allows more NIR)")
        print("  full_spectrum  - Full spectrum (no filtering)")
        return
    
    directory = args.directory
    ir_filter_type = args.ir_filter
    
    print(f"Loading spectral bands from: {directory}")
    print(f"Using IR cut-off filter: {ir_filter_type}")
    
    # Load all spectral bands
    all_bands = load_spectral_bands(directory)
    
    if all_bands is None:
        print("Failed to load spectral bands.")
        return
    
    print(f"Loaded {len(all_bands)} spectral bands")
    print("Available wavelengths:", [band["wavelength"] for band in all_bands])
    
    # Create spectral response curves with specified IR filter
    print(f"\nCreating spectral response curves with {ir_filter_type} IR filter...")
    response_curves = create_spectral_response_curves(ir_filter_type)
    
    # Display response curves
    display_spectral_response_curves(response_curves)
    
    # Simulate different CFA patterns
    print("\nSimulating CFA patterns...")
    cfa_results = {}
    
    # Bayer (RGGB)
    bayer_rgb, bayer_channels = simulate_bayer_cfa(all_bands, response_curves)
    cfa_results["Bayer (RGGB)"] = (bayer_rgb, bayer_channels)
    
    # RCCB
    rccb_rgb, rccb_channels = simulate_rccb_cfa(all_bands, response_curves)
    cfa_results["RCCB"] = (rccb_rgb, rccb_channels)
    
    # RCCG
    rccg_rgb, rccg_channels = simulate_rccg_cfa(all_bands, response_curves)
    cfa_results["RCCG"] = (rccg_rgb, rccg_channels)
    
    # RYYCy
    ryycy_rgb, ryycy_channels = simulate_ryycy_cfa(all_bands, response_curves)
    cfa_results["RYYCy"] = (ryycy_rgb, ryycy_channels)
    
    # Create comparison visualization
    print("\nCreating CFA comparison...")
    comparison_path = f"cfa_comparison_{ir_filter_type}.png"
    create_cfa_comparison(cfa_results, comparison_path)
    
    # Save individual images
    print("\nSaving CFA simulation images...")
    output_dir = f"cfa_simulations_{ir_filter_type}"
    saved_files = save_cfa_images(cfa_results, output_dir)
    
    print(f"\nCFA simulation complete!")
    print(f"Generated {len(cfa_results)} CFA patterns with {ir_filter_type} IR filter")
    print(f"Saved {len(saved_files)} images to {output_dir}/")


if __name__ == "__main__":
    main()
