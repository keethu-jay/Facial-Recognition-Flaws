# H2: Epsilon Sensitivity Analysis

This folder contains the analysis for testing whether demographic disparities change with perturbation magnitude (epsilon).

## Files

- **H2_Epsilon_Sensitivity.py** - Main analysis script
- **H2_epsilon_sensitivity_data.csv** - ASR data for extreme racial groups across epsilon values
- **H2_Epsilon_Sensitivity.png** - Line plot showing ASR vs epsilon for most/least vulnerable races

## Running the Analysis

```bash
cd hypothesis_testing/H2_Epsilon_Sensitivity
python H2_Epsilon_Sensitivity.py
```

## Results

The analysis:
1. Identifies the most and least vulnerable racial groups from baseline data
2. Compares their ASR across different epsilon values (0.01, 0.03, 0.05)
3. Determines if disparities widen, narrow, or remain constant with increasing perturbation budget

This helps understand if larger perturbations exacerbate or mitigate existing demographic disparities.




