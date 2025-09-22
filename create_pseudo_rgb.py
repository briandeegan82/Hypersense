#!/usr/bin/env python3
"""
Pseudo-RGB Image Creator from Multi-Spectral Bands
Allows users to select any 3 of the 14 spectral bands to create custom pseudo-RGB images.

Requirements:
- numpy: pip install numpy
- matplotlib: pip install matplotlib
- opencv-python: pip install opencv-python

Usage:
    python create_pseudo_rgb.py [path_to_cropped_images_directory]
    or
    python create_pseudo_rgb.py  # will look for cropped_images_DSC00204/ directory
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
                "position": info["name"],
                "index": len(all_bands)
            })
        
        if bands["G"] is not None:
            all_bands.append({
                "wavelength": bands["G"],
                "channel": "G", 
                "data": image[:, :, 1],  # Green channel
                "position": info["name"],
                "index": len(all_bands)
            })
        
        if bands["B"] is not None:
            all_bands.append({
                "wavelength": bands["B"],
                "channel": "B",
                "data": image[:, :, 2],  # Blue channel
                "position": info["name"],
                "index": len(all_bands)
            })
    
    # Sort by wavelength
    all_bands.sort(key=lambda x: x["wavelength"])
    
    # Reassign indices after sorting
    for i, band in enumerate(all_bands):
        band["index"] = i
    
    return all_bands


def display_available_bands(all_bands):
    """Display all available spectral bands for selection."""
    print("\nAvailable Spectral Bands:")
    print("=" * 60)
    for i, band in enumerate(all_bands):
        print(f"{i+1:2d}. {band['wavelength']:3d}nm ({band['channel']}) from {band['position'].replace('_', ' ').title()}")
    print("=" * 60)


def select_bands(all_bands):
    """Allow user to select 3 bands for pseudo-RGB creation."""
    display_available_bands(all_bands)
    
    print("\nSelect 3 bands for pseudo-RGB image:")
    print("Enter band numbers (1-14) separated by spaces or commas")
    print("Example: 1 5 10  or  1,5,10")
    
    while True:
        try:
            selection = input("\nYour selection: ").strip()
            
            # Parse input (handle both spaces and commas)
            if ',' in selection:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
            else:
                indices = [int(x.strip()) - 1 for x in selection.split()]
            
            # Validate selection
            if len(indices) != 3:
                print("Please select exactly 3 bands.")
                continue
                
            if any(i < 0 or i >= len(all_bands) for i in indices):
                print(f"Please enter numbers between 1 and {len(all_bands)}.")
                continue
            
            # Check for duplicates
            if len(set(indices)) != 3:
                print("Please select 3 different bands.")
                continue
            
            selected_bands = [all_bands[i] for i in indices]
            
            print(f"\nSelected bands:")
            for i, band in enumerate(selected_bands):
                print(f"  {['Red', 'Green', 'Blue'][i]}: {band['wavelength']}nm ({band['channel']}) from {band['position'].replace('_', ' ').title()}")
            
            confirm = input("\nProceed with this selection? (y/n): ").strip().lower()
            if confirm in ['y', 'yes']:
                return selected_bands
            else:
                print("Selection cancelled. Please try again.")
                
        except ValueError:
            print("Invalid input. Please enter numbers only.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None


def create_pseudo_rgb(selected_bands, output_path="pseudo_rgb.png"):
    """Create a pseudo-RGB image from the selected bands."""
    if len(selected_bands) != 3:
        raise ValueError("Exactly 3 bands are required for RGB creation")
    
    # Get the shape of the first band (all should be the same after alignment)
    height, width = selected_bands[0]["data"].shape
    
    # Create RGB image
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Assign bands to R, G, B channels
    rgb_image[:, :, 0] = selected_bands[0]["data"]  # Red channel
    rgb_image[:, :, 1] = selected_bands[1]["data"]  # Green channel
    rgb_image[:, :, 2] = selected_bands[2]["data"]  # Blue channel
    
    return rgb_image


def display_pseudo_rgb(rgb_image, selected_bands, output_path="pseudo_rgb.png"):
    """Display and save the pseudo-RGB image."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Display the pseudo-RGB image
    axes[0].imshow(rgb_image)
    axes[0].set_title("Pseudo-RGB Image", fontsize=14)
    axes[0].axis('off')
    
    # Display individual bands
    for i, band in enumerate(selected_bands):
        axes[1].imshow(band["data"], cmap='gray')
        axes[1].set_title(f"Band {i+1}: {band['wavelength']}nm ({band['channel']})", fontsize=12)
        axes[1].axis('off')
        break  # Only show first band as example
    
    # Create title with band information
    band_info = []
    for i, band in enumerate(selected_bands):
        color = ['Red', 'Green', 'Blue'][i]
        band_info.append(f"{color}: {band['wavelength']}nm")
    
    title = "Pseudo-RGB Composition\n" + " | ".join(band_info)
    plt.suptitle(title, fontsize=14, y=0.95)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Pseudo-RGB image saved: {output_path}")
    
    # Also save just the RGB image
    rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    rgb_only_path = output_path.replace('.png', '_rgb_only.png')
    cv2.imwrite(rgb_only_path, rgb_bgr)
    print(f"RGB-only image saved: {rgb_only_path}")


def create_multiple_combinations(all_bands, output_dir="pseudo_rgb_combinations"):
    """Create several common pseudo-RGB combinations."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Define some common combinations
    combinations = [
        {
            "name": "Natural Color",
            "description": "Red (650nm), Green (525nm), Blue (430nm)",
            "wavelengths": [650, 525, 430]
        },
        {
            "name": "False Color NIR",
            "description": "NIR (850nm), Red (650nm), Green (525nm)",
            "wavelengths": [850, 650, 525]
        },
        {
            "name": "Vegetation Analysis",
            "description": "NIR (850nm), Red Edge (710nm), Red (650nm)",
            "wavelengths": [850, 710, 650]
        },
        {
            "name": "Blue-Green-Red",
            "description": "Blue (405nm), Green (525nm), Red (650nm)",
            "wavelengths": [405, 525, 650]
        },
        {
            "name": "Red Edge Analysis",
            "description": "Red Edge (735nm), Red (685nm), Green (560nm)",
            "wavelengths": [735, 685, 560]
        }
    ]
    
    print(f"\nCreating {len(combinations)} common pseudo-RGB combinations...")
    
    for combo in combinations:
        try:
            # Find bands with matching wavelengths
            selected_bands = []
            for wavelength in combo["wavelengths"]:
                matching_bands = [b for b in all_bands if b["wavelength"] == wavelength]
                if matching_bands:
                    selected_bands.append(matching_bands[0])  # Take first match
                else:
                    print(f"Warning: No band found for {wavelength}nm in {combo['name']}")
                    break
            
            if len(selected_bands) == 3:
                # Create pseudo-RGB
                rgb_image = create_pseudo_rgb(selected_bands)
                
                # Save
                filename = f"{combo['name'].replace(' ', '_').lower()}.png"
                filepath = os.path.join(output_dir, filename)
                
                # Convert to BGR for OpenCV
                rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(filepath, rgb_bgr)
                
                print(f"  ✓ {combo['name']}: {filepath}")
            else:
                print(f"  ✗ {combo['name']}: Could not find all required bands")
                
        except Exception as e:
            print(f"  ✗ {combo['name']}: Error - {e}")
    
    print(f"\nCommon combinations saved in: {output_dir}")


def main():
    """Main function to handle command line arguments and create pseudo-RGB images."""
    if len(sys.argv) > 1:
        # If a specific directory is provided
        directory = sys.argv[1]
    else:
        # Look for cropped images directory
        directory = "cropped_images_DSC00204"
    
    print(f"Loading spectral bands from: {directory}")
    
    # Load all spectral bands
    all_bands = load_spectral_bands(directory)
    
    if all_bands is None:
        print("Failed to load spectral bands.")
        return
    
    print(f"Loaded {len(all_bands)} spectral bands")
    
    while True:
        print("\n" + "="*60)
        print("PSEUDO-RGB IMAGE CREATOR")
        print("="*60)
        print("1. Create custom pseudo-RGB image")
        print("2. Create common pseudo-RGB combinations")
        print("3. Display available bands")
        print("4. Exit")
        
        try:
            choice = input("\nSelect an option (1-4): ").strip()
            
            if choice == "1":
                # Custom pseudo-RGB creation
                selected_bands = select_bands(all_bands)
                if selected_bands:
                    rgb_image = create_pseudo_rgb(selected_bands)
                    display_pseudo_rgb(rgb_image, selected_bands)
                    
            elif choice == "2":
                # Create common combinations
                create_multiple_combinations(all_bands)
                
            elif choice == "3":
                # Display available bands
                display_available_bands(all_bands)
                
            elif choice == "4":
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please select 1-4.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
