# How to Run Adversarial Data Generation

## Important: Run from the Correct Directory

The `generate_adv_data.py` script **must be run from the `Datasets/` directory** because:
1. It needs to find `FaceNet_Model.py` in the parent directory
2. It needs to access the baseline data in `baseline/` subdirectory
3. It needs to access the race dataset `.pkl` files in the current directory

## Method 1: Command Line (Recommended)

```bash
# Navigate to Datasets directory
cd Datasets

# Run the script
python generate_adv_data.py
```

## Method 2: Using the Batch Script (Windows)

Double-click `run_generate_adv_data.bat` or run:
```bash
cd Datasets
run_generate_adv_data.bat
```

## Method 3: From Project Root

If you're in the project root directory, you can run:
```bash
cd Datasets && python generate_adv_data.py
```

## Verification

Before running, verify:
1. You're in the `Datasets/` directory
2. `FaceNet_Model.py` exists in the parent directory
3. `baseline/all_races_pairs.csv` exists
4. Race dataset files (e.g., `East_Asian.pkl`) exist in `Datasets/`

## Expected Output

You should see:
```
**********************************************************************
ADVERSARIAL DATA GENERATION
**********************************************************************

Loading FaceNet model...
[OK] FaceNet model loaded successfully using keras-facenet library.

Loading baseline pairs...
[OK] Loaded 1234 baseline pairs

======================================================================
Generating adversarial data: FGSM eps=0.01
======================================================================
```

If you see an error about FaceNet model not found, check:
- Are you in the `Datasets/` directory?
- Does `../FaceNet_Model.py` exist?
- Is `keras-facenet` installed? (`pip install keras-facenet`)

## Time Estimate

- **Total time**: 3-8 hours
- **Per attack/epsilon**: ~20-30 minutes
- **Total combinations**: 9 (3 attacks × 3 epsilons)
- Progress is shown every 100 pairs

## Output Location

Results will be saved to:
- `outputs/1_Adversarial_Data/FGSM_e0.01_data.csv`
- `outputs/1_Adversarial_Data/FGSM_e0.03_data.csv`
- ... (9 files total)
- `outputs/1_Adversarial_Data/all_adversarial_data.csv` (combined)

