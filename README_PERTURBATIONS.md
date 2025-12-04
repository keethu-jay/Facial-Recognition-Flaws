# Creating Perturbed Datasets with White-Box Attacks

This guide explains how to create three versions of each demographic dataset using white-box adversarial attacks (FGSM, PGD, C&W).

## Overview

The script `Datasets/CreatePerturbedDatasets.py` uses **FaceNet model gradients** to create adversarial examples. These are **white-box attacks** that require access to the model's internal gradients.

**Result**: For each of the 7 races, you'll have:
- Original dataset: `{Race}.pkl` (e.g., `Black.pkl`)
- C&W perturbed: `{Race}_CW.pkl` (e.g., `Black_CW.pkl`)
- PGD perturbed: `{Race}_PGD.pkl` (e.g., `Black_PGD.pkl`)
- FGSM perturbed: `{Race}_FGSM.pkl` (e.g., `Black_FGSM.pkl`)

## Prerequisites

### 1. Install Dependencies

```bash
pip install tensorflow pillow numpy
pip install keras-facenet
```

**OR** if you prefer to use the original David Sandberg FaceNet:
- Download FaceNet weights from: https://github.com/davidsandberg/facenet
- Place the model file in a `models/` directory

### 2. Project Structure

```
Facial-Recognition-Flaws/
├── Datasets/
│   ├── CreatePerturbedDatasets.py  # Attack generation script
│   ├── DatasetCreation.py           # Original dataset creation
│   ├── Black.pkl                    # Original datasets
│   ├── White.pkl
│   └── ...
├── FaceNet_Model.py                 # FaceNet model loader
├── models/                           # (Optional) For custom weights
│   └── facenet_weights.h5
└── README_PERTURBATIONS.md
```

## Usage

### Basic Usage

```bash
cd Datasets
python CreatePerturbedDatasets.py
```

The script will:
1. Load the FaceNet model
2. For each race dataset:
   - Load the original images
   - Pair each image with another from the same gender/age group
   - Apply FGSM, PGD, and C&W attacks
   - Save the perturbed datasets

## Understanding the Attacks

### FGSM (Fast Gradient Sign Method)

**Single-step attack** that adds noise in the direction of the gradient.

**Key Code Lines:**
```python
# Compute gradient of loss with respect to input image
gradient = tape.gradient(loss, image_tensor)

# Take sign (direction only, not magnitude)
signed_gradient = tf.sign(gradient)

# HERE IS WHERE THE NOISE IS ADDED
perturbed_image = image_tensor + epsilon * signed_gradient
```

**Where epsilon is added**: Line `perturbed_image = image_tensor + epsilon * signed_gradient`
- `epsilon` controls how much noise is added
- Typical values: 0.01 (subtle) to 0.1 (more visible)

### PGD (Projected Gradient Descent)

**Iterative attack** that takes many small steps, projecting back to epsilon-ball.

**Key Code Lines:**
```python
# Take a small step (alpha) in gradient direction
perturbed_image = perturbed_image + alpha * signed_gradient

# HERE IS WHERE WE PROJECT BACK TO EPSILON-BALL
delta = perturbed_image - original_image
delta = tf.clip_by_value(delta, -epsilon, epsilon)
perturbed_image = original_image + delta
```

**Where epsilon is used**: 
- `alpha` (step size) is typically `epsilon/10`
- Final projection ensures total change ≤ `epsilon`

### C&W (Carlini & Wagner)

**Optimization-based attack** that finds minimum perturbation.

**Key Code Lines:**
```python
# Transform w to image space (keeps values in [0,1])
x_adv = (tf.tanh(w) + 1.0) / 2.0

# Calculate total loss: perturbation_size + c * attack_penalty
total_loss = perturbation_norm + c * attack_loss

# Optimizer adjusts w to minimize loss
optimizer.apply_gradients([(gradients, w)])
```

**Where noise is added**: Through optimization - the optimizer finds the minimum perturbation needed.

## Testing Workflow

1. **Create perturbed datasets** (this script)
   ```bash
   python Datasets/CreatePerturbedDatasets.py
   ```

2. **Create benchmark pairs**
   - Pair faces from same race, gender, and age range
   - Example: Two Asian females aged 20-29 from `East_Asian.pkl`

3. **Test baseline with FaceNet**
   - Compare original faces in pairs
   - FaceNet should correctly identify them as different (distance > 1.0)

4. **Test with perturbed versions**
   - Replace one face in each pair with its perturbed version
   - Test if FaceNet still correctly identifies them as different
   - If FaceNet misidentifies them (distance < 1.0), the attack succeeded

## Parameters

You can adjust attack parameters in `main()`:

```python
# Perturbation magnitude (for FGSM and PGD)
epsilon = 0.01  # Smaller = less visible, larger = more visible

# PGD-specific
alpha = 0.001   # Step size (typically epsilon/10)
num_iter = 10   # Number of iterations

# C&W-specific
learning_rate = 0.01
num_iter = 100
binary_search_steps = 9
```

## Notes

- **White-box attacks require FaceNet model** - the script needs gradients from the model
- Processing can take time (especially C&W which is optimization-based)
- The perturbed images maintain the same structure as originals
- All metadata (race, gender, age) is preserved
- Each perturbed image includes `target_index` showing which image it was targeted to match

## Troubleshooting

### "Could not load FaceNet model"

**Solution 1**: Install keras-facenet
```bash
pip install keras-facenet
```

**Solution 2**: Download weights manually
1. Visit: https://github.com/davidsandberg/facenet
2. Download pre-trained weights (e.g., `20180402-114759` model)
3. Place in `models/` directory
4. Update `FaceNet_Model.py` to point to the correct file

### "Out of memory" errors

- Reduce batch size or process one image at a time
- Use smaller epsilon values
- Process one race at a time

