# Creating Perturbed Datasets - Usage Guide

This guide explains how to create three versions of each demographic dataset with different perturbation types (C&W, PGD, FGSM).

## Overview

The script `CreatePerturbedDatasets.py` will:
1. Load each of the 7 demographic datasets (Black, White, East_Asian, etc.)
2. Apply each of the 3 perturbation types (C&W, PGD, FGSM) to each face in the dataset
3. Save the perturbed datasets as new .pkl files

**Result**: For each of the 7 races, you'll have:
- Original dataset: `{Race}.pkl` (e.g., `Black.pkl`)
- C&W perturbed: `{Race}_CW.pkl` (e.g., `Black_CW.pkl`)
- PGD perturbed: `{Race}_PGD.pkl` (e.g., `Black_PGD.pkl`)
- FGSM perturbed: `{Race}_FGSM.pkl` (e.g., `Black_FGSM.pkl`)

**Note**: This script does NOT require FaceNet. It applies perturbations directly to images. You can test the perturbed datasets with FaceNet later.

## Prerequisites

1. **Install required packages**:
   ```bash
   pip install pillow numpy
   ```

2. **Have your original datasets ready**: 
   - The original .pkl files should be in the `Datasets/` directory
   - Files: `Black.pkl`, `White.pkl`, `East_Asian.pkl`, etc.

## Usage

### Basic Usage

Simply run the script:

```bash
cd Datasets
python CreatePerturbedDatasets.py
```

The script will:
- Load each original dataset
- Apply all three perturbation types to each image
- Save the perturbed datasets

**No FaceNet model is required** - perturbations are applied directly to images.

### Custom Usage

You can also use the functions programmatically:

```python
from CreatePerturbedDatasets import (
    load_dataset, 
    create_perturbed_dataset, 
    save_perturbed_dataset
)

# Create perturbed dataset for a specific race and perturbation type
perturbed_dataset = create_perturbed_dataset('Black', 'FGSM')
save_perturbed_dataset(perturbed_dataset, 'Black', 'FGSM')
```

## Perturbation Types

The script applies simplified versions of these perturbation types:

### FGSM (Fast Gradient Sign Method)
- Simulates FGSM-style structured noise
- Parameters: `epsilon` (default: 0.01) - controls perturbation magnitude

### PGD (Projected Gradient Descent)
- Iterative perturbation with projection to epsilon-ball
- Parameters: 
  - `epsilon` (default: 0.01) - maximum perturbation
  - `num_iter` (default: 10) - number of iterations

### C&W (Carlini & Wagner)
- Smooth, optimization-style perturbation
- Parameters: `epsilon` (default: 0.01) - perturbation magnitude

**Note**: These are simplified perturbations applied directly to images. For true adversarial attacks, you would need FaceNet to compute gradients. However, these perturbations will still alter the images and can be tested with FaceNet to see their effect.

## Output Structure

After running the script, you'll have:

```
Datasets/
├── Black.pkl              # Original
├── Black_CW.pkl           # C&W perturbed
├── Black_PGD.pkl          # PGD perturbed
├── Black_FGSM.pkl         # FGSM perturbed
├── White.pkl
├── White_CW.pkl
├── White_PGD.pkl
├── White_FGSM.pkl
└── ... (same for all 7 races)
```

## Loading Perturbed Datasets

```python
import pickle

# Load a perturbed dataset
with open('Datasets/Black_FGSM.pkl', 'rb') as f:
    dataset = pickle.load(f)

race_name = list(dataset.keys())[0]  # 'Black'
images = dataset[race_name]  # List of image data dictionaries

# Each image has:
# - 'image': PIL Image (perturbed)
# - 'perturbation_type': 'FGSM', 'PGD', or 'CW'
# - 'race', 'gender', 'age': Original metadata
# - Other original metadata fields
```

## Customizing Attack Parameters

To modify attack parameters, edit the functions in `CreatePerturbedDatasets.py`:

```python
# In apply_perturbation function, modify:
if perturbation_type == 'FGSM':
    perturbed_tensor = fgsm_attack(model, image_tensor, target_image_tensor, epsilon=0.02)  # Changed epsilon
```

## Testing Workflow

1. **Create perturbed datasets** (this script)
   - Run `CreatePerturbedDatasets.py` to generate all perturbed versions

2. **Create benchmark pairs**
   - Pair faces from same race, gender, and age range
   - Example: Two Asian females aged 20-29

3. **Test baseline with FaceNet**
   - Compare original faces in pairs
   - FaceNet should correctly identify them as different (distance > 1.0)

4. **Test with perturbed versions**
   - Replace one face in each pair with its perturbed version
   - Test if FaceNet still correctly identifies them as different
   - If FaceNet misidentifies them (distance < 1.0), the perturbation succeeded

## Notes

- The script processes images sequentially, which may take some time
- The perturbed images maintain the same structure as originals, just with modified pixel values
- All metadata (race, gender, age) is preserved in the perturbed datasets
- **No FaceNet model is required** to create the perturbations - you only need FaceNet for testing later

