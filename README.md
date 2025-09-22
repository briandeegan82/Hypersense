# Multi-Spectral ARW Image Processing Toolkit

A comprehensive Python toolkit for processing, aligning, and analyzing multi-spectral images from Agrowing cameras that capture 14 spectral bands in a 2×3 layout within a single ARW file.

## Installation

1. Clone or download this repository
2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install rawpy matplotlib numpy opencv-python scipy
```

3. Activate the virtual environment (if using one):
```bash
source venv/bin/activate
```

## Features

### 1. ARW File Viewer (`display_arw.py`)
- Opens and displays Sony .ARW RAW files
- Shows camera information and image metadata
- Displays image dimensions and processing information
- Handles both single file and batch processing
- User-friendly interface with error handling

### 2. Multi-Spectral Image Cropping and Alignment (`crop_align_arw.py`)
- **Extracts 6 spectral images** from 2×3 layout in ARW files
- **15% border cropping** to avoid neighboring image contamination
- **Automatic alignment** using SIFT feature detection and matching
- **Top-middle image as reference** for consistent alignment
- **14 individual spectral bands** extracted and labeled
- **Spectral band visualization** with wavelength information
- **Comparison grids** showing original vs aligned images

### 3. Pseudo-RGB Image Creator (`create_pseudo_rgb.py`)
- **Interactive band selection** from 14 available spectral bands
- **Pre-defined combinations** for common applications
- **Custom RGB creation** using any 3 spectral bands
- **Spectral band visualization** with wavelength labels
- **High-resolution output** with proper color channel assignment

### 4. CFA Pattern Simulator (`simulate_cfa_patterns.py`)
- **4 CFA pattern simulations**: Bayer (RGGB), RCCB, RCCG, RYYCy
- **Configurable IR cut-off filters**: Standard, Sharp, Extended, No Filter
- **Realistic spectral response curves** for each color channel
- **Weighted spectral combinations** based on filter characteristics
- **Comprehensive visualization** of filter effects and color responses

## Spectral Band Information

The multi-spectral camera captures **14 spectral channels** across a 2×3 layout:

| Position | Red Channel | Green Channel | Blue Channel |
|----------|-------------|---------------|--------------|
| Top Left | 850nm (NIR) | - | - |
| Top Middle | 650nm | 525nm | - |
| Top Right | 710nm | 570nm | 405nm |
| Bottom Left | 650nm | 550nm | 430nm |
| Bottom Middle | 735nm | - | 490nm |
| Bottom Right | 685nm | 560nm | 450nm |

**Spectral Coverage:**
- **Blue/Violet**: 405nm, 430nm, 450nm, 490nm
- **Green**: 525nm, 550nm, 560nm, 570nm  
- **Red/NIR**: 650nm, 685nm, 710nm, 735nm, 850nm

## Usage

### 1. Display ARW Files

#### Display a specific ARW file:
```bash
python display_arw.py path/to/your/file.ARW
```

#### Display ARW files from input_images directory:
```bash
python display_arw.py
```

#### Using convenience script:
```bash
./run_arw_viewer.sh path/to/your/file.ARW
```

### 2. Crop and Align Multi-Spectral Images

#### Process a specific ARW file:
```bash
python crop_align_arw.py input_images/DSC00204.ARW
```

#### Process all ARW files in input_images directory:
```bash
python crop_align_arw.py
```

#### Using convenience script:
```bash
./run_crop_align.sh input_images/DSC00204.ARW
```

**Output:**
- `cropped_images_[filename]/` directory with:
  - 6 aligned spectral images
  - Individual spectral bands plot (14 bands)
  - Spectral bands display (6 composite images)
  - Comparison grid (original vs aligned)

### 3. Create Pseudo-RGB Images

#### Interactive mode:
```bash
python create_pseudo_rgb.py
```

#### Process specific directory:
```bash
python create_pseudo_rgb.py cropped_images_DSC00204
```

#### Using convenience script:
```bash
./run_pseudo_rgb.sh
```

**Menu Options:**
1. **Create custom pseudo-RGB image** - Select any 3 of 14 spectral bands
2. **Create common pseudo-RGB combinations** - Pre-defined useful combinations
3. **Display available bands** - Show all 14 bands with wavelengths
4. **Exit**

**Pre-defined Combinations:**
- **Natural Color**: Red (650nm), Green (525nm), Blue (430nm)
- **False Color NIR**: NIR (850nm), Red (650nm), Green (525nm)
- **Vegetation Analysis**: NIR (850nm), Red Edge (710nm), Red (650nm)
- **Blue-Green-Red**: Blue (405nm), Green (525nm), Red (650nm)
- **Red Edge Analysis**: Red Edge (735nm), Red (685nm), Green (560nm)

### 4. Simulate CFA Patterns

#### List available IR filters:
```bash
python simulate_cfa_patterns.py --list-filters
```

#### Simulate with specific IR filter:
```bash
python simulate_cfa_patterns.py --ir-filter standard_ir
```

#### Process specific directory with custom filter:
```bash
python simulate_cfa_patterns.py cropped_images_DSC00204 --ir-filter extended_ir
```

#### Using convenience script:
```bash
./run_cfa_simulator.sh --ir-filter sharp_ir
```

**Available IR Filters:**
- `no_filter` - No IR cut-off (passes all wavelengths)
- `standard_ir` - Standard IR cut-off at ~650nm (typical consumer cameras)
- `sharp_ir` - Sharp IR cut-off at ~700nm (professional cameras)
- `extended_ir` - Extended IR cut-off at ~800nm (allows more NIR)
- `full_spectrum` - Full spectrum (no filtering)

**CFA Patterns:**
- **Bayer (RGGB)** - Traditional Red, Green, Green, Blue pattern
- **RCCB** - Red, Clear, Clear, Blue pattern
- **RCCG** - Red, Clear, Clear, Green pattern
- **RYYCy** - Red, Yellow, Yellow, Cyan pattern

## Output Files

### Cropping and Alignment Results
- `cropped_images_[filename]/` directory containing:
  - 6 aligned spectral images (top_left, top_middle, top_right, bottom_left, bottom_middle, bottom_right)
  - `[filename]_individual_bands.png` - All 14 individual spectral bands
  - `[filename]_spectral_bands.png` - 6 composite images with band labels
  - `[filename]_comparison.png` - Original vs aligned comparison grid

### Pseudo-RGB Results
- `pseudo_rgb_combinations/` directory containing:
  - 5 pre-defined pseudo-RGB combinations
  - Custom combinations (when created interactively)

### CFA Simulation Results
- `cfa_simulations_[filter_type]/` directories containing:
  - 4 CFA pattern RGB composites
  - Individual channel images for each CFA pattern
- `cfa_comparison_[filter_type].png` - Side-by-side comparison of all CFA patterns
- `spectral_response_curves.png` - IR filter and color channel response curves

## Scientific Applications

### Vegetation Analysis
- **NIR combinations** reveal plant health and biomass
- **Red edge analysis** shows vegetation stress
- **Chlorophyll mapping** using specific wavelength combinations

### Material Classification
- **Different band combinations** highlight different materials
- **Spectral signatures** for material identification
- **Multi-spectral classification** algorithms

### Camera Design
- **CFA pattern comparison** for optimal sensor design
- **IR filter effects** on image quality
- **Spectral response optimization** for specific applications

### Research Applications
- **Hyperspectral data analysis** through standard CFA patterns
- **Color filter optimization** studies
- **Multi-spectral data visualization** and interpretation

## Requirements

- **Python 3.6+**
- **rawpy** - RAW file processing
- **matplotlib** - Image display and visualization
- **numpy** - Numerical operations
- **opencv-python** - Image alignment algorithms
- **scipy** - Optimization and signal processing

## Technical Notes

### Alignment Algorithm
- Uses **SIFT feature detection** for robust alignment
- **FLANN-based matching** for efficient feature correspondence
- **RANSAC homography estimation** for geometric transformation
- **Top-middle image as reference** for consistent results

### Spectral Processing
- **15% border cropping** prevents neighboring image contamination
- **Weighted spectral combinations** based on realistic filter responses
- **Normalized responses** ensure proper color balance
- **High-fidelity simulation** of camera sensor behavior

### Quality Assurance
- **Feature match validation** with minimum threshold requirements
- **Graceful fallback** to original images if alignment fails
- **Comprehensive error handling** and user feedback
- **High-resolution output** (150 DPI) for analysis

## Examples

### Complete Processing Pipeline
```bash
# 1. Process ARW file through complete pipeline
python crop_align_arw.py input_images/DSC00204.ARW

# 2. Create pseudo-RGB combinations
python create_pseudo_rgb.py cropped_images_DSC00204

# 3. Simulate CFA patterns with different IR filters
python simulate_cfa_patterns.py cropped_images_DSC00204 --ir-filter standard_ir
python simulate_cfa_patterns.py cropped_images_DSC00204 --ir-filter extended_ir
python simulate_cfa_patterns.py cropped_images_DSC00204 --ir-filter no_filter
```

### Batch Processing
```bash
# Process all ARW files in input_images directory
for file in input_images/*.ARW; do
    echo "Processing $file"
    python crop_align_arw.py "$file"
done
```

## License

This project is open source and available under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## Support

For questions or support, please open an issue in the repository or contact the development team.