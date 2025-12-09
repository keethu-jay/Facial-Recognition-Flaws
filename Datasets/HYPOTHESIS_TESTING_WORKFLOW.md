# Hypothesis Testing Workflow Guide

This guide explains how to run the complete hypothesis testing pipeline to answer your research questions.

## Prerequisites

### 1. Install Required Packages

```bash
pip install pandas numpy matplotlib seaborn scipy statsmodels tensorflow keras-facenet opencv-python pillow
```

### 2. Required Data Files

Make sure you have:
- ✅ Baseline statistics created (`baseline/all_races_pairs.csv`)
- ✅ FaceNet model accessible (will load automatically)

## Workflow Overview

The complete workflow consists of 4 main steps:

```
1. Generate Adversarial Data (3-8 hours)
   ↓
2. Run Hypothesis Tests (H1-H4)
   ↓
3. Create Final Visualizations
   ↓
4. Review Report
```

## Step-by-Step Instructions

### Option 1: Run Everything Automatically

```bash
cd Datasets
python run_all_hypothesis_tests.py
```

This master script will:
- Check if adversarial data exists
- Generate it if needed (with your confirmation)
- Run all hypothesis tests (H1-H4)
- Create all visualizations
- Generate the report

### Option 2: Run Steps Manually

#### Step 1: Generate Adversarial Data

**Time Required**: 3-8 hours (depending on hardware)

```bash
cd Datasets
python generate_adv_data.py
```

**What it does**:
- Loads baseline pairs from `baseline/all_races_pairs.csv`
- For each pair, perturbs Image A to minimize distance to Image B
- Tests 3 attacks × 3 epsilons = 9 combinations
- Saves results to `outputs/1_Adversarial_Data/`

**Output Files**:
- `FGSM_e0.01_data.csv`
- `FGSM_e0.03_data.csv`
- `FGSM_e0.05_data.csv`
- `PGD_e0.01_data.csv`
- `PGD_e0.03_data.csv`
- `PGD_e0.05_data.csv`
- `CW_e0.01_data.csv`
- `CW_e0.03_data.csv`
- `CW_e0.05_data.csv`
- `all_adversarial_data.csv` (combined)

#### Step 2: Run Hypothesis Tests

Run each hypothesis test individually:

```bash
# H1: Demographic Disparity (ANOVA)
python H1_ASR_ANOVA.py

# H2: Epsilon Sensitivity
python H2_Epsilon_Sensitivity.py

# H3: Compounding Harm
python H3_Compounding_Harm.py

# H4: Attack Consistency
python H4_Attack_Consistency.py
```

**Or run all at once**:
```bash
python H1_ASR_ANOVA.py && python H2_Epsilon_Sensitivity.py && python H3_Compounding_Harm.py && python H4_Attack_Consistency.py
```

#### Step 3: Create Final Visualizations

```bash
python create_final_visualizations.py
```

This creates:
- Baseline FAR heatmap
- Adversarial ASR heatmap

#### Step 4: Review Report

Open `analysis_report.md` to see all results compiled together.

## Output Structure

After running all scripts, you'll have:

```
Datasets/
├── outputs/
│   ├── 1_Adversarial_Data/
│   │   ├── FGSM_e0.01_data.csv
│   │   ├── FGSM_e0.03_data.csv
│   │   ├── ... (9 total files)
│   │   └── all_adversarial_data.csv
│   ├── 2_Hypothesis_Stats/
│   │   ├── H1_ANOVA_PGD_e0.03.csv
│   │   ├── H1_aggregated_ASR_PGD_e0.03.csv
│   │   ├── H2_epsilon_sensitivity_data.csv
│   │   ├── H3_compounding_harm_comparison.csv
│   │   ├── H3_compounding_harm_summary.txt
│   │   └── H4_attack_consistency.csv
│   └── 3_Final_Visuals/
│       ├── H1_ASR_Bar_Chart.png
│       ├── H2_Epsilon_Sensitivity.png
│       ├── H3_Compounding_Harm.png
│       ├── H4_Attack_Consistency.png
│       ├── Baseline_FAR_Heatmap.png
│       └── Adversarial_ASR_Heatmap.png
└── analysis_report.md
```

## What Each Hypothesis Tests

### H1: Demographic Disparity
- **Question**: Are there significant differences in ASR across demographics?
- **Method**: Three-Way ANOVA (Race × Gender × Age)
- **Output**: ANOVA table, ASR bar chart

### H2: Epsilon Sensitivity
- **Question**: Do disparities change with perturbation magnitude?
- **Method**: Compare extreme races across epsilon values
- **Output**: Line plot showing ASR vs epsilon

### H3: Compounding Harm
- **Question**: Do attacks compound existing disparities?
- **Method**: Compare baseline FAR ratio to adversarial ASR ratio
- **Output**: Comparison table, side-by-side bar charts

### H4: Attack Consistency
- **Question**: Do different attacks show consistent disparities?
- **Method**: Compare disparity ratios across attack types
- **Output**: Bar chart comparing attack methods

## Troubleshooting

### "ModuleNotFoundError: No module named 'statsmodels'"
```bash
pip install statsmodels
```

### "Adversarial data not found"
Run `generate_adv_data.py` first. This takes 3-8 hours.

### "Baseline pairs not found"
Run `CreateBaselineStats.py` first to generate baseline data.

### Scripts take too long
- Adversarial data generation is the bottleneck (3-8 hours)
- Hypothesis tests are fast (minutes)
- Consider running adversarial generation overnight

## Quick Start (If Data Already Exists)

If you already have adversarial data:

```bash
cd Datasets
python run_all_hypothesis_tests.py
```

The script will detect existing data and skip generation.

