#!/usr/bin/env python3
"""
ARW Image Cropping and Alignment Script
Crops and aligns 6 RGB images from a 2x3 layout in Sony .ARW files.
Uses the top middle image as the reference for alignment.

Requirements:
- rawpy: pip install rawpy
- opencv-python: pip install opencv-python
- numpy: pip install numpy
- matplotlib: pip install matplotlib

Usage:
    python crop_align_arw.py [path_to_arw_file]
    or
    python crop_align_arw.py  # will look for ARW files in input_images/ directory
"""

import os
import sys
import glob
import rawpy
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path


def find_arw_files(directory="input_images"):
    """Find all ARW files in the specified directory."""
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found.")
        return []
    
    arw_files = glob.glob(os.path.join(directory, "*.ARW")) + glob.glob(os.path.join(directory, "*.arw"))
    return arw_files


def crop_2x3_images(rgb_image, border_percent=0.1):
    """
    Crop the 2x3 layout of multi-spectral images from the full ARW image with a border to avoid neighboring image data.
    Returns a list of 6 cropped images with their spectral band information.
    
    Spectral channels:
    - Top left: R=850nm, G=null, B=null
    - Top middle: R=650nm, G=525nm, B=null  
    - Top right: R=710nm, G=570nm, B=405nm
    - Bottom left: R=650nm, G=550nm, B=430nm
    - Bottom middle: R=735nm, G=null, B=490nm
    - Bottom right: R=685nm, G=560nm, B=450nm
    
    Args:
        rgb_image: The full RGB image from the ARW file
        border_percent: Percentage of border to add around each image (default 0.1 = 10%)
    """
    height, width = rgb_image.shape[:2]
    
    # Calculate dimensions for each sub-image
    sub_height = height // 2
    sub_width = width // 3
    
    # Calculate border sizes
    border_y = int(sub_height * border_percent)
    border_x = int(sub_width * border_percent)
    
    cropped_images = []
    
    # Define spectral band information for each position
    spectral_info = [
        {"name": "top_left", "position": (0, 0), "bands": {"R": 850, "G": None, "B": None}},
        {"name": "top_middle", "position": (0, sub_width), "bands": {"R": 650, "G": 525, "B": None}},
        {"name": "top_right", "position": (0, 2 * sub_width), "bands": {"R": 710, "G": 570, "B": 405}},
        {"name": "bottom_left", "position": (sub_height, 0), "bands": {"R": 650, "G": 550, "B": 430}},
        {"name": "bottom_middle", "position": (sub_height, sub_width), "bands": {"R": 735, "G": None, "B": 490}},
        {"name": "bottom_right", "position": (sub_height, 2 * sub_width), "bands": {"R": 685, "G": 560, "B": 450}}
    ]
    
    for info in spectral_info:
        y_start, x_start = info["position"]
        # Add border to the crop region
        y_start_bordered = max(0, y_start + border_y)
        x_start_bordered = max(0, x_start + border_x)
        y_end_bordered = min(height, y_start + sub_height - border_y)
        x_end_bordered = min(width, x_start + sub_width - border_x)
        
        cropped = rgb_image[y_start_bordered:y_end_bordered, x_start_bordered:x_end_bordered]
        cropped_images.append({
            "image": cropped,
            "name": info["name"],
            "bands": info["bands"]
        })
    
    return cropped_images


def align_images(images, reference_idx=1):
    """
    Align all images to the reference image (top-middle by default).
    Uses feature detection and matching for alignment.
    """
    if len(images) != 6:
        raise ValueError("Expected 6 images for 2x3 layout")
    
    reference_image = images[reference_idx]["image"]
    aligned_images = [None] * 6
    aligned_images[reference_idx] = {
        "image": reference_image.copy(),
        "name": images[reference_idx]["name"],
        "bands": images[reference_idx]["bands"]
    }
    
    # Convert reference to grayscale for feature detection
    ref_gray = cv2.cvtColor(reference_image, cv2.COLOR_RGB2GRAY)
    
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # Find keypoints and descriptors for reference image
    ref_kp, ref_des = sift.detectAndCompute(ref_gray, None)
    
    # Create FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    print(f"Reference image (index {reference_idx}) has {len(ref_kp)} keypoints")
    
    for i, image_data in enumerate(images):
        if i == reference_idx:
            continue
            
        print(f"Aligning image {i} ({image_data['name']})...")
        
        # Convert to grayscale
        gray = cv2.cvtColor(image_data["image"], cv2.COLOR_RGB2GRAY)
        
        # Find keypoints and descriptors
        kp, des = sift.detectAndCompute(gray, None)
        
        if des is None or len(des) < 10:
            print(f"Warning: Image {i} has insufficient features ({len(kp) if kp else 0} keypoints)")
            aligned_images[i] = {
                "image": image_data["image"].copy(),
                "name": image_data["name"],
                "bands": image_data["bands"]
            }
            continue
        
        # Match descriptors
        matches = flann.knnMatch(ref_des, des, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        print(f"Image {i}: {len(good_matches)} good matches from {len(matches)} total matches")
        
        if len(good_matches) < 10:
            print(f"Warning: Image {i} has insufficient good matches, using original")
            aligned_images[i] = {
                "image": image_data["image"].copy(),
                "name": image_data["name"],
                "bands": image_data["bands"]
            }
            continue
        
        # Extract matched keypoints
        ref_pts = np.float32([ref_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        img_pts = np.float32([kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography matrix
        try:
            homography, mask = cv2.findHomography(img_pts, ref_pts, cv2.RANSAC, 5.0)
            
            if homography is not None:
                # Apply homography to align the image
                h, w = reference_image.shape[:2]
                aligned = cv2.warpPerspective(image_data["image"], homography, (w, h))
                aligned_images[i] = {
                    "image": aligned,
                    "name": image_data["name"],
                    "bands": image_data["bands"]
                }
                print(f"Image {i} aligned successfully")
            else:
                print(f"Warning: Could not find homography for image {i}, using original")
                aligned_images[i] = {
                    "image": image_data["image"].copy(),
                    "name": image_data["name"],
                    "bands": image_data["bands"]
                }
                
        except Exception as e:
            print(f"Error aligning image {i}: {e}, using original")
            aligned_images[i] = {
                "image": image_data["image"].copy(),
                "name": image_data["name"],
                "bands": image_data["bands"]
            }
    
    return aligned_images


def save_cropped_images(images, output_dir="cropped_images", base_name="cropped"):
    """Save the cropped and aligned images to files."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    saved_files = []
    
    for i, image_data in enumerate(images):
        filename = f"{base_name}_{image_data['name']}.png"
        filepath = os.path.join(output_dir, filename)
        
        # Convert RGB to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_data["image"], cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, image_bgr)
        saved_files.append(filepath)
        print(f"Saved: {filepath}")
    
    return saved_files


def extract_individual_bands(images):
    """Extract all individual spectral bands from the aligned images."""
    all_bands = []
    
    for image_data in images:
        bands = image_data["bands"]
        image = image_data["image"]
        
        # Extract R, G, B channels if they exist
        if bands["R"] is not None:
            all_bands.append({
                "wavelength": bands["R"],
                "channel": "R",
                "data": image[:, :, 0],  # Red channel
                "position": image_data["name"]
            })
        
        if bands["G"] is not None:
            all_bands.append({
                "wavelength": bands["G"],
                "channel": "G", 
                "data": image[:, :, 1],  # Green channel
                "position": image_data["name"]
            })
        
        if bands["B"] is not None:
            all_bands.append({
                "wavelength": bands["B"],
                "channel": "B",
                "data": image[:, :, 2],  # Blue channel
                "position": image_data["name"]
            })
    
    # Sort by wavelength
    all_bands.sort(key=lambda x: x["wavelength"])
    return all_bands


def create_individual_bands_plot(images, output_path="individual_spectral_bands.png"):
    """Create a comprehensive plot showing all 14 individual spectral bands."""
    all_bands = extract_individual_bands(images)
    
    # Create a grid layout for 14 bands
    n_bands = len(all_bands)
    n_cols = 4  # 4 columns
    n_rows = (n_bands + n_cols - 1) // n_cols  # Calculate rows needed
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    
    # Flatten axes for easier indexing
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes_flat = axes.flatten()
    
    for i, band in enumerate(all_bands):
        ax = axes_flat[i]
        
        # Display the band as grayscale
        im = ax.imshow(band["data"], cmap='gray')
        
        # Create title with wavelength and position info
        title = f"{band['wavelength']}nm ({band['channel']})\n{band['position'].replace('_', ' ').title()}"
        ax.set_title(title, fontsize=10, pad=5)
        ax.axis('off')
        
        # Add colorbar for each band
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Hide unused subplots
    for i in range(n_bands, len(axes_flat)):
        axes_flat[i].axis('off')
    
    plt.suptitle(f"All {n_bands} Individual Spectral Bands - Aligned", fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Individual spectral bands plot saved: {output_path}")
    
    # Print summary of all bands
    print(f"\nExtracted {n_bands} spectral bands:")
    for band in all_bands:
        print(f"  {band['wavelength']}nm ({band['channel']}) from {band['position']}")
    
    return all_bands


def create_spectral_display(images, output_path="spectral_bands.png"):
    """Create a spectral band display showing all 6 multi-spectral images with their wavelengths."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, image_data in enumerate(images):
        # Create title with spectral information
        bands = image_data["bands"]
        band_info = []
        if bands["R"] is not None:
            band_info.append(f"R: {bands['R']}nm")
        if bands["G"] is not None:
            band_info.append(f"G: {bands['G']}nm")
        if bands["B"] is not None:
            band_info.append(f"B: {bands['B']}nm")
        
        title = f"{image_data['name'].replace('_', ' ').title()}\n" + " | ".join(band_info)
        
        axes[i].imshow(image_data["image"])
        axes[i].set_title(title, fontsize=10, pad=10)
        axes[i].axis('off')
    
    plt.suptitle("Multi-Spectral Camera Bands - Aligned Images", fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Spectral bands display saved: {output_path}")


def create_comparison_grid(original_images, aligned_images, output_path="comparison_grid.png"):
    """Create a comparison grid showing original vs aligned images."""
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    for i in range(6):
        # Original images (top row)
        axes[0, i].imshow(original_images[i]["image"])
        axes[0, i].set_title(f"Original - {original_images[i]['name'].replace('_', ' ').title()}")
        axes[0, i].axis('off')
        
        # Aligned images (bottom row)
        axes[1, i].imshow(aligned_images[i]["image"])
        axes[1, i].set_title(f"Aligned - {aligned_images[i]['name'].replace('_', ' ').title()}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Comparison grid saved: {output_path}")


def process_arw_file(file_path):
    """Process a single ARW file to crop and align the 6 RGB images."""
    try:
        print(f"Processing ARW file: {file_path}")
        
        # Read the RAW file
        with rawpy.imread(file_path) as raw:
            print(f"Image size: {raw.sizes.width} x {raw.sizes.height}")
            
            # Post-process the raw image to obtain an RGB image
            rgb_image = raw.postprocess(
                use_camera_wb=True,  # Use camera white balance
                half_size=False,     # Full resolution
                no_auto_bright=True, # Don't auto-adjust brightness
                output_bps=8         # 8-bit output
            )
            
            print(f"Processed image shape: {rgb_image.shape}")
            
            # Crop the 2x3 images with 15% border
            print("Cropping 2x3 images with 15% border...")
            cropped_images = crop_2x3_images(rgb_image, border_percent=0.15)
            
            print(f"Cropped {len(cropped_images)} images")
            for i, img_data in enumerate(cropped_images):
                print(f"  Image {i} ({img_data['name']}): {img_data['image'].shape}")
                bands = img_data['bands']
                band_str = []
                if bands['R']: band_str.append(f"R:{bands['R']}nm")
                if bands['G']: band_str.append(f"G:{bands['G']}nm")
                if bands['B']: band_str.append(f"B:{bands['B']}nm")
                print(f"    Bands: {' | '.join(band_str)}")
            
            # Align images using top-middle as reference
            print("Aligning images...")
            aligned_images = align_images(cropped_images, reference_idx=1)
            
            # Create output directory based on input filename
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_dir = f"cropped_images_{base_name}"
            
            # Save cropped and aligned images
            print("Saving images...")
            saved_files = save_cropped_images(aligned_images, output_dir, base_name)
            
            # Create spectral bands display
            spectral_path = os.path.join(output_dir, f"{base_name}_spectral_bands.png")
            create_spectral_display(aligned_images, spectral_path)
            
            # Create individual bands plot
            individual_bands_path = os.path.join(output_dir, f"{base_name}_individual_bands.png")
            create_individual_bands_plot(aligned_images, individual_bands_path)
            
            # Create comparison grid
            comparison_path = os.path.join(output_dir, f"{base_name}_comparison.png")
            create_comparison_grid(cropped_images, aligned_images, comparison_path)
            
            print(f"\nProcessing complete!")
            print(f"Output directory: {output_dir}")
            print(f"Saved {len(saved_files)} aligned images")
            
            return True
            
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return False


def main():
    """Main function to handle command line arguments and process ARW files."""
    if len(sys.argv) > 1:
        # If a specific file path is provided
        file_path = sys.argv[1]
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        
        if not file_path.lower().endswith(('.arw', '.ARW')):
            print("Please provide an ARW file.")
            return
            
        process_arw_file(file_path)
    else:
        # Look for ARW files in the input_images directory
        arw_files = find_arw_files()
        
        if not arw_files:
            print("No ARW files found in 'input_images/' directory.")
            print("Usage: python crop_align_arw.py [path_to_arw_file]")
            return
        
        print(f"Found {len(arw_files)} ARW file(s):")
        for i, file_path in enumerate(arw_files, 1):
            print(f"{i}. {file_path}")
        
        if len(arw_files) == 1:
            # If only one file, process it directly
            process_arw_file(arw_files[0])
        else:
            # If multiple files, let user choose
            try:
                choice = input(f"\nEnter the number of the file to process (1-{len(arw_files)}): ")
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(arw_files):
                    process_arw_file(arw_files[choice_idx])
                else:
                    print("Invalid choice.")
            except (ValueError, KeyboardInterrupt):
                print("Operation cancelled.")


if __name__ == "__main__":
    main()
