# Guide: Creating Perturbed Datasets

This guide explains how to create perturbed versions of your demographic datasets using white-box adversarial attacks.

## Prerequisites

### 1. Install Required Packages

```bash
pip install tensorflow keras-facenet opencv-python pillow numpy pandas
```

### 2. FaceNet Model

The script uses the **keras-facenet** library which automatically downloads FaceNet weights on first use. See `MODEL_DOCUMENTATION.md` for details about the model source.

**Model Source**: 
- Library: `keras-facenet` (open-source)
- Pre-trained weights: Automatically downloaded from the library's repository
- Architecture: Inception ResNet v1 trained on VGGFace2
- See `MODEL_DOCUMENTATION.md` for full attribution

### 3. Original Datasets

Make sure you have the original race datasets in the `Datasets/` folder:
- `Black.pkl`
- `White.pkl`
- `East_Asian.pkl`
- `Indian.pkl`
- `Middle_Eastern.pkl`
- `Latino_Hispanic.pkl`
- `Southeast_Asian.pkl`

## Running the Script

### Basic Usage

```bash
cd Datasets
python CreatePerturbedDatasets.py
```

### What It Does

The script will:

1. **Load FaceNet model** (downloads weights automatically on first run)
2. **Process each race** (7 races total)
3. **Apply 3 attack types** to each race:
   - FGSM (Fast Gradient Sign Method)
   - PGD (Projected Gradient Descent)
   - C&W (Carlini & Wagner)
4. **Test 3 epsilon values** for each attack:
   - ε = 0.01
   - ε = 0.03
   - ε = 0.05
5. **Save results** in organized folder structure

### Output Structure

The script creates the following folder structure:

```
Datasets/
├── FGSM/
│   ├── Black/
│   │   ├── Black_0.01.pkl
│   │   ├── Black_0.03.pkl
│   │   └── Black_0.05.pkl
│   ├── White/
│   │   ├── White_0.01.pkl
│   │   ├── White_0.03.pkl
│   │   └── White_0.05.pkl
│   └── ... (5 more races)
├── PGD/
│   └── ... (same structure)
└── CW/
    └── ... (same structure)
```

**Total datasets created**: 7 races × 3 attacks × 3 epsilons = **63 perturbed datasets**

## How It Works

### 1. Pairing Strategy

For each image in a dataset:
- Finds another image from the **same gender/age group** (different person)
- Creates a pair: (original_image, target_image)
- Goal: Make FaceNet think these two different people are the same

### 2. Attack Process

Each attack type works differently:

#### FGSM (Fast Gradient Sign Method)
- **Single-step attack**
- Computes gradient of loss with respect to input
- Adds epsilon-scaled noise in gradient direction
- Formula: `x_adv = x + ε * sign(∇_x J)`

#### PGD (Projected Gradient Descent)
- **Iterative attack** (10 iterations)
- Takes small steps (alpha = epsilon/10) per iteration
- Projects back to epsilon-ball after each step
- More powerful than FGSM

#### C&W (Carlini & Wagner)
- **Optimization-based attack**
- Finds minimum perturbation needed
- Uses L∞ constraint (epsilon) for consistency
- Most sophisticated attack

### 3. Epsilon Values

Three perturbation budgets are tested:
- **ε = 0.01**: Subtle perturbation (barely visible)
- **ε = 0.03**: Moderate perturbation (somewhat visible)
- **ε = 0.05**: More noticeable perturbation

These values represent the **L∞ norm** (maximum pixel change) allowed.

## Time Requirements

**Warning**: This process takes a **very long time**!

- **Per image**: ~2-5 seconds (depending on attack type)
- **Per race**: ~90 images × 3 attacks × 3 epsilons = 810 image processings
- **Total**: ~7 races × 810 = ~5,670 image processings
- **Estimated time**: 3-8 hours (depending on your hardware)

### Tips for Faster Processing

1. **Process one race at a time**: Modify the script to process only one race
2. **Use GPU**: If available, TensorFlow will automatically use it
3. **Process overnight**: Let it run while you sleep
4. **Test with one epsilon first**: Modify `EPSILON_VALUES` to test with just one value

## Verifying Results

After completion, check:

1. **Folder structure**: All 63 datasets should be created
2. **File sizes**: Each .pkl file should be ~10-15 MB
3. **Metadata**: Each dataset preserves original metadata (race, gender, age)

## Using Perturbed Datasets

The perturbed datasets can be used with the baseline pairs:

1. **Load baseline pairs**: `pairs_df = pd.read_csv('baseline/Black_pairs.csv')`
2. **Load perturbed dataset**: `perturbed_data = pickle.load(open('FGSM/Black/Black_0.01.pkl', 'rb'))`
3. **Test same pairs**: Use `image_A_index` and `image_B_index` from pairs
4. **Compare distances**: Compare baseline vs perturbed distances

## Troubleshooting

### "Could not load FaceNet model"
- Install: `pip install keras-facenet opencv-python`
- Check internet connection (needed for first-time weight download)

### "Out of memory"
- Process one race at a time
- Reduce batch processing
- Close other applications

### "Takes too long"
- This is normal - the process is computationally intensive
- Consider processing overnight
- Use GPU if available

## Model Attribution

**Important**: When documenting your work, cite:

1. **FaceNet Model**: 
   - Library: keras-facenet (open-source)
   - See `MODEL_DOCUMENTATION.md` for full details

2. **Attack Methods**:
   - FGSM: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2014)
   - PGD: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2017)
   - C&W: Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks" (2017)

## Next Steps

After creating perturbed datasets:

1. **Test with baseline pairs**: Use the same pairs from baseline to compare results
2. **Calculate attack success rates**: Measure how often attacks succeed
3. **Compare demographic disparities**: Analyze if attacks affect different groups differently
4. **Generate comparison visualizations**: Create charts comparing baseline vs perturbed performance

