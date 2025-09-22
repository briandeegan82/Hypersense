#!/usr/bin/env python3
"""
Multispectral Image Alignment Script for Agrowing Camera
A Python script to extract and align 6 spectral bands from ARW files captured with an Agrowing camera.

The Agrowing camera uses a 6-element lens system to capture multispectral images.
Each spectral band is captured as a separate region of the sensor.

Requirements:
- rawpy: pip install rawpy
- matplotlib: pip install matplotlib
- numpy: pip install numpy
- opencv-python: pip install opencv-python
- scipy: pip install scipy

Usage:
    python align_multispectral.py [path_to_arw_file]
    or
    python align_multispectral.py  # will look for ARW files in input_images/ directory
"""

import os
import sys
import glob
import rawpy
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy import ndimage
from scipy.optimize import minimize
from pathlib import Path
import argparse


class MultispectralAligner:
    def __init__(self, arw_file_path):
        """Initialize the multispectral aligner with an ARW file."""
        self.arw_file_path = arw_file_path
        self.raw_image = None
        self.spectral_bands = None
        self.aligned_bands = None
        self.image_height = None
        self.image_width = None
        
        # Agrowing camera wavelength information (based on documentation)
        self.wavelength_info = {
            '4_band_ndvi': {
                'bands': 4,
                'wavelengths': [650, 550, 450, 850],  # R, G, B, NIR
                'names': ['Red', 'Green', 'Blue', 'NIR'],
                'layout': '2x2'
            },
            '4_band_rededge': {
                'bands': 4,
                'wavelengths': [710, 550, 450, 850],  # RedEdge, G, B, NIR
                'names': ['RedEdge', 'Green', 'Blue', 'NIR'],
                'layout': '2x2'
            },
            '14_band_sextuple': {
                'bands': 14,
                'wavelengths': [850, 550, 430, 685, 560, 450, 710, 570, 405, 650, 630, 525, 735, 490],  # Based on documentation
                'names': ['NIR1', 'Green1', 'Blue1', 'Red1', 'Green2', 'Blue2', 'RedEdge1', 'Green3', 'Blue3', 'Red2', 'Red3', 'Green4', 'RedEdge2', 'Blue4'],
                'layout': '2x7'
            },
            '10_band_v2': {
                'bands': 10,
                'wavelengths': [850, 550, 430, 685, 560, 450, 710, 570, 405, 650],
                'names': ['NIR', 'Green1', 'Blue1', 'Red1', 'Green2', 'Blue2', 'RedEdge1', 'Green3', 'Blue3', 'Red2'],
                'layout': '2x5'
            }
        }
        
    def load_raw_image(self):
        """Load the raw ARW image."""
        try:
            print(f"Loading ARW file: {self.arw_file_path}")
            
            with rawpy.imread(self.arw_file_path) as raw:
                # Get raw image data without demosaicing
                self.raw_image = raw.raw_image_visible.copy()
                self.image_height, self.image_width = self.raw_image.shape
                
                print(f"Raw image size: {self.image_width} x {self.image_height}")
                print(f"Raw image data type: {self.raw_image.dtype}")
                print(f"Raw image value range: {self.raw_image.min()} - {self.raw_image.max()}")
                
                return True
                
        except Exception as e:
            print(f"Error loading ARW file: {str(e)}")
            return False
    
    def extract_spectral_bands(self, band_layout='auto'):
        """
        Extract 6 spectral bands from the raw image.
        
        For a 6-element lens system, the bands are typically arranged in a grid pattern.
        Common layouts:
        - 2x3 grid (2 rows, 3 columns)
        - 3x2 grid (3 rows, 2 columns)
        - 1x6 horizontal strip
        - 6x1 vertical strip
        """
        if self.raw_image is None:
            print("Error: Raw image not loaded. Call load_raw_image() first.")
            return False
        
        print(f"Extracting spectral bands with layout: {band_layout}")
        
        # Try to automatically detect the layout based on image dimensions
        if band_layout == 'auto':
            # For a 9568 x 6376 image, a 2x3 layout seems most likely
            # Each band would be approximately 3188 x 4784 pixels
            band_layout = '2x3'
        
        self.spectral_bands = []
        
        if band_layout == '2x3':
            # 2 rows, 3 columns
            band_height = self.image_height // 2
            band_width = self.image_width // 3
            
            for row in range(2):
                for col in range(3):
                    y_start = row * band_height
                    y_end = (row + 1) * band_height
                    x_start = col * band_width
                    x_end = (col + 1) * band_width
                    
                    band = self.raw_image[y_start:y_end, x_start:x_end]
                    self.spectral_bands.append(band)
                    print(f"Band {len(self.spectral_bands)}: {band.shape} (region: {y_start}:{y_end}, {x_start}:{x_end})")
        
        elif band_layout == '3x2':
            # 3 rows, 2 columns
            band_height = self.image_height // 3
            band_width = self.image_width // 2
            
            for row in range(3):
                for col in range(2):
                    y_start = row * band_height
                    y_end = (row + 1) * band_height
                    x_start = col * band_width
                    x_end = (col + 1) * band_width
                    
                    band = self.raw_image[y_start:y_end, x_start:x_end]
                    self.spectral_bands.append(band)
                    print(f"Band {len(self.spectral_bands)}: {band.shape} (region: {y_start}:{y_end}, {x_start}:{x_end})")
        
        elif band_layout == '1x6':
            # 1 row, 6 columns (horizontal strip)
            band_height = self.image_height
            band_width = self.image_width // 6
            
            for col in range(6):
                x_start = col * band_width
                x_end = (col + 1) * band_width
                
                band = self.raw_image[:, x_start:x_end]
                self.spectral_bands.append(band)
                print(f"Band {len(self.spectral_bands)}: {band.shape} (region: :, {x_start}:{x_end})")
        
        elif band_layout == '6x1':
            # 6 rows, 1 column (vertical strip)
            band_height = self.image_height // 6
            band_width = self.image_width
            
            for row in range(6):
                y_start = row * band_height
                y_end = (row + 1) * band_height
                
                band = self.raw_image[y_start:y_end, :]
                self.spectral_bands.append(band)
                print(f"Band {len(self.spectral_bands)}: {band.shape} (region: {y_start}:{y_end}, :)")
        
        elif band_layout == '2x2':
            # 2 rows, 2 columns (4 bands)
            band_height = self.image_height // 2
            band_width = self.image_width // 2
            
            for row in range(2):
                for col in range(2):
                    y_start = row * band_height
                    y_end = (row + 1) * band_height
                    x_start = col * band_width
                    x_end = (col + 1) * band_width
                    
                    band = self.raw_image[y_start:y_end, x_start:x_end]
                    self.spectral_bands.append(band)
                    print(f"Band {len(self.spectral_bands)}: {band.shape} (region: {y_start}:{y_end}, {x_start}:{x_end})")
        
        else:
            print(f"Unknown band layout: {band_layout}")
            return False
        
        print(f"Successfully extracted {len(self.spectral_bands)} spectral bands")
        return True
    
    def detect_camera_config(self):
        """
        Automatically detect the camera configuration based on image dimensions.
        """
        if self.raw_image is None:
            print("Error: Raw image not loaded.")
            return None
        
        width, height = self.image_width, self.image_height
        print(f"Detecting camera configuration for image size: {width} x {height}")
        
        # Based on documentation, common configurations:
        # 4-band: 2516x3976 per band
        # 6-band: various sizes
        # 10-band: 3544x2316 per band
        
        # Calculate potential band dimensions
        potential_configs = []
        
        # Check for 6-band configurations (2x3 layout) - prioritize sextuple
        # Allow for slight dimension mismatches (common in real cameras)
        if height % 2 == 0:
            band_w, band_h = width // 3, height // 2
            # Check if the division is reasonable (within 1 pixel tolerance)
            if abs(width - band_w * 3) <= 1:
                potential_configs.append({
                    'config': '6_band_sextuple',
                    'bands': 6,
                    'layout': '2x3',
                    'band_size': (band_w, band_h),
                    'confidence': 0.95  # High confidence for sextuple
                })
        
        # Check for 6-band configurations (3x2 layout)
        if width % 2 == 0 and height % 3 == 0:
            band_w, band_h = width // 2, height // 3
            potential_configs.append({
                'config': '6_band_sextuple',
                'bands': 6,
                'layout': '3x2',
                'band_size': (band_w, band_h),
                'confidence': 0.9
            })
        
        # Check for 6-band configurations (1x6 layout)
        if width % 6 == 0:
            band_w, band_h = width // 6, height
            potential_configs.append({
                'config': '6_band_sextuple',
                'bands': 6,
                'layout': '1x6',
                'band_size': (band_w, band_h),
                'confidence': 0.8
            })
        
        # Check for 6-band configurations (6x1 layout)
        if height % 6 == 0:
            band_w, band_h = width, height // 6
            potential_configs.append({
                'config': '6_band_sextuple',
                'bands': 6,
                'layout': '6x1',
                'band_size': (band_w, band_h),
                'confidence': 0.8
            })
        
        # Check for 4-band configurations (2x2 layout)
        if width % 2 == 0 and height % 2 == 0:
            band_w, band_h = width // 2, height // 2
            potential_configs.append({
                'config': '4_band_ndvi',
                'bands': 4,
                'layout': '2x2',
                'band_size': (band_w, band_h),
                'confidence': 0.7
            })
        
        # Check for 10-band configurations (2x5 layout)
        if width % 5 == 0 and height % 2 == 0:
            band_w, band_h = width // 5, height // 2
            potential_configs.append({
                'config': '10_band_v2',
                'bands': 10,
                'layout': '2x5',
                'band_size': (band_w, band_h),
                'confidence': 0.6
            })
        
        # Select the most likely configuration
        if potential_configs:
            best_config = max(potential_configs, key=lambda x: x['confidence'])
            print(f"Detected configuration: {best_config['config']}")
            print(f"  Layout: {best_config['layout']}")
            print(f"  Band size: {best_config['band_size']}")
            print(f"  Confidence: {best_config['confidence']}")
            return best_config
        else:
            print("Could not automatically detect camera configuration")
            return None
    
    def normalize_band(self, band):
        """Normalize a spectral band to 0-255 range."""
        band_min = band.min()
        band_max = band.max()
        if band_max > band_min:
            normalized = ((band - band_min) / (band_max - band_min) * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(band, dtype=np.uint8)
        return normalized
    
    def find_best_alignment(self, reference_band, target_band, max_shift=50):
        """
        Find the best alignment between two bands using cross-correlation.
        Returns the optimal (dx, dy) shift.
        """
        # Normalize bands for better correlation
        ref_norm = self.normalize_band(reference_band)
        target_norm = self.normalize_band(target_band)
        
        # Use a smaller region for correlation to speed up computation
        # Take center region of both bands
        h, w = ref_norm.shape
        center_h, center_w = h // 2, w // 2
        region_size = min(1000, h // 4, w // 4)  # Use smaller region for correlation
        
        ref_region = ref_norm[center_h-region_size//2:center_h+region_size//2,
                             center_w-region_size//2:center_w+region_size//2]
        target_region = target_norm[center_h-region_size//2:center_h+region_size//2,
                                   center_w-region_size//2:center_w+region_size//2]
        
        # Perform cross-correlation
        correlation = cv2.matchTemplate(target_region, ref_region, cv2.TM_CCOEFF_NORMED)
        
        # Find the best match
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(correlation)
        
        # Calculate the shift
        dx = max_loc[0] - region_size // 2
        dy = max_loc[1] - region_size // 2
        
        # Limit the shift to reasonable bounds
        dx = np.clip(dx, -max_shift, max_shift)
        dy = np.clip(dy, -max_shift, max_shift)
        
        return dx, dy, max_val
    
    def align_bands(self, reference_band_idx=0):
        """
        Align all spectral bands to a reference band.
        """
        if self.spectral_bands is None or len(self.spectral_bands) == 0:
            print("Error: No spectral bands extracted. Call extract_spectral_bands() first.")
            return False
        
        print(f"Aligning bands to reference band {reference_band_idx}")
        
        reference_band = self.spectral_bands[reference_band_idx]
        self.aligned_bands = [reference_band.copy()]
        
        # Calculate the common size for all aligned bands
        # We'll crop all bands to the smallest size after alignment
        min_height = reference_band.shape[0]
        min_width = reference_band.shape[1]
        
        shifts = [(0, 0)]  # Reference band has no shift
        
        for i, band in enumerate(self.spectral_bands):
            if i == reference_band_idx:
                continue
            
            print(f"Aligning band {i} to reference band {reference_band_idx}")
            dx, dy, correlation_score = self.find_best_alignment(reference_band, band)
            shifts.append((dx, dy))
            
            print(f"  Shift: dx={dx}, dy={dy}, correlation={correlation_score:.3f}")
            
            # Apply the shift
            if dx != 0 or dy != 0:
                # Create transformation matrix
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned_band = cv2.warpAffine(band.astype(np.float32), M, 
                                            (band.shape[1], band.shape[0]))
            else:
                aligned_band = band.copy()
            
            self.aligned_bands.append(aligned_band)
            
            # Update minimum dimensions
            min_height = min(min_height, aligned_band.shape[0])
            min_width = min(min_width, aligned_band.shape[1])
        
        # Crop all bands to the same size
        print(f"Cropping all bands to common size: {min_width} x {min_height}")
        for i in range(len(self.aligned_bands)):
            h, w = self.aligned_bands[i].shape
            y_start = (h - min_height) // 2
            x_start = (w - min_width) // 2
            self.aligned_bands[i] = self.aligned_bands[i][y_start:y_start+min_height,
                                                         x_start:x_start+min_width]
        
        print(f"Successfully aligned {len(self.aligned_bands)} spectral bands")
        return True
    
    def visualize_bands(self, show_original=True, show_aligned=True, save_images=False, config_name=None):
        """
        Visualize the spectral bands before and after alignment with wavelength information.
        """
        if self.spectral_bands is None:
            print("Error: No spectral bands extracted.")
            return
        
        num_bands = len(self.spectral_bands)
        
        # Get wavelength information if available
        wavelength_info = None
        if config_name and config_name in self.wavelength_info:
            wavelength_info = self.wavelength_info[config_name]
        
        # Create figure with subplots
        if show_original and show_aligned:
            fig, axes = plt.subplots(2, num_bands, figsize=(3*num_bands, 6))
            fig.suptitle('Spectral Bands: Original (top) vs Aligned (bottom)', fontsize=16)
        elif show_original:
            fig, axes = plt.subplots(1, num_bands, figsize=(3*num_bands, 3))
            fig.suptitle('Original Spectral Bands', fontsize=16)
        else:
            fig, axes = plt.subplots(1, num_bands, figsize=(3*num_bands, 3))
            fig.suptitle('Aligned Spectral Bands', fontsize=16)
        
        # Ensure axes is 2D
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_bands):
            if i < len(self.spectral_bands):
                # Create title with wavelength info if available
                title = f'Band {i+1}'
                if wavelength_info and i < len(wavelength_info['wavelengths']):
                    wavelength = wavelength_info['wavelengths'][i]
                    name = wavelength_info['names'][i] if i < len(wavelength_info['names']) else f'Band{i+1}'
                    title = f'{name}\n{wavelength}nm'
                
                if show_original:
                    # Show original band
                    band_norm = self.normalize_band(self.spectral_bands[i])
                    axes[0, i].imshow(band_norm, cmap='gray')
                    axes[0, i].set_title(f'{title} (Original)', fontsize=10)
                    axes[0, i].axis('off')
                
                if show_aligned and self.aligned_bands is not None and i < len(self.aligned_bands):
                    # Show aligned band
                    band_norm = self.normalize_band(self.aligned_bands[i])
                    row_idx = 1 if show_original else 0
                    axes[row_idx, i].imshow(band_norm, cmap='gray')
                    axes[row_idx, i].set_title(f'{title} (Aligned)', fontsize=10)
                    axes[row_idx, i].axis('off')
            else:
                # Empty subplot
                for row in range(axes.shape[0]):
                    axes[row, i].axis('off')
        
        plt.tight_layout()
        
        if save_images:
            output_path = f"spectral_bands_{os.path.basename(self.arw_file_path).split('.')[0]}.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {output_path}")
        
        plt.show()
    
    def save_aligned_bands(self, output_dir="aligned_bands"):
        """
        Save the aligned spectral bands as individual images.
        """
        if self.aligned_bands is None:
            print("Error: No aligned bands available.")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.basename(self.arw_file_path).split('.')[0]
        
        for i, band in enumerate(self.aligned_bands):
            # Normalize band for saving
            band_norm = self.normalize_band(band)
            
            # Save as PNG
            output_path = os.path.join(output_dir, f"{base_name}_band_{i+1}.png")
            cv2.imwrite(output_path, band_norm)
            print(f"Saved band {i+1} to: {output_path}")
        
        return True


def find_arw_files(directory="input_images"):
    """Find all ARW files in the specified directory."""
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found.")
        return []
    
    arw_files = glob.glob(os.path.join(directory, "*.ARW")) + glob.glob(os.path.join(directory, "*.arw"))
    return arw_files


def main():
    """Main function to handle command line arguments and process ARW files."""
    parser = argparse.ArgumentParser(description='Align multispectral images from Agrowing camera')
    parser.add_argument('arw_file', nargs='?', help='Path to ARW file to process')
    parser.add_argument('--layout', choices=['auto', '2x3', '3x2', '1x6', '6x1'], 
                       default='auto', help='Band layout pattern')
    parser.add_argument('--reference', type=int, default=0, 
                       help='Reference band index for alignment (0-5)')
    parser.add_argument('--save', action='store_true', 
                       help='Save aligned bands as individual images')
    parser.add_argument('--no-display', action='store_true', 
                       help='Skip displaying the visualization')
    
    args = parser.parse_args()
    
    # Determine which ARW file to process
    if args.arw_file:
        if not os.path.exists(args.arw_file):
            print(f"File not found: {args.arw_file}")
            return
        
        if not args.arw_file.lower().endswith(('.arw', '.ARW')):
            print("Please provide an ARW file.")
            return
        
        arw_files = [args.arw_file]
    else:
        # Look for ARW files in the input_images directory
        arw_files = find_arw_files()
        
        if not arw_files:
            print("No ARW files found in 'input_images/' directory.")
            print("Usage: python align_multispectral.py [path_to_arw_file]")
            return
        
        print(f"Found {len(arw_files)} ARW file(s):")
        for i, file_path in enumerate(arw_files, 1):
            print(f"{i}. {file_path}")
        
        if len(arw_files) == 1:
            arw_files = arw_files
        else:
            # If multiple files, let user choose
            try:
                choice = input(f"\nEnter the number of the file to process (1-{len(arw_files)}): ")
                choice_idx = int(choice) - 1
                
                if 0 <= choice_idx < len(arw_files):
                    arw_files = [arw_files[choice_idx]]
                else:
                    print("Invalid choice.")
                    return
            except (ValueError, KeyboardInterrupt):
                print("Operation cancelled.")
                return
    
    # Process the selected ARW file
    for arw_file in arw_files:
        print(f"\n{'='*60}")
        print(f"Processing: {arw_file}")
        print(f"{'='*60}")
        
        # Create aligner instance
        aligner = MultispectralAligner(arw_file)
        
        # Load raw image
        if not aligner.load_raw_image():
            continue
        
        # Auto-detect camera configuration if layout is 'auto'
        detected_config = None
        if args.layout == 'auto':
            detected_config = aligner.detect_camera_config()
            if detected_config:
                args.layout = detected_config['layout']
                config_name = detected_config['config']
            else:
                print("Auto-detection failed, using default 2x3 layout")
                args.layout = '2x3'
                config_name = '6_band_sextuple'
        else:
            # Map layout to config name
            layout_to_config = {
                '2x2': '4_band_ndvi',
                '2x3': '6_band_sextuple',
                '2x5': '10_band_v2'
            }
            config_name = layout_to_config.get(args.layout, '6_band_sextuple')
        
        # Extract spectral bands
        if not aligner.extract_spectral_bands(band_layout=args.layout):
            continue
        
        # Align bands
        if not aligner.align_bands(reference_band_idx=args.reference):
            continue
        
        # Visualize results with wavelength information
        if not args.no_display:
            aligner.visualize_bands(save_images=True, config_name=config_name)
        
        # Save aligned bands if requested
        if args.save:
            aligner.save_aligned_bands()
        
        print(f"\nProcessing completed for: {arw_file}")


if __name__ == "__main__":
    main()
