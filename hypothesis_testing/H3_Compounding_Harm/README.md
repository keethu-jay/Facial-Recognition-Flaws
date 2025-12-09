# H3: Compounding Harm Analysis

This folder contains the analysis for testing whether adversarial attacks compound existing demographic disparities.

## Files

- **H3_Compounding_Harm.py** - Main analysis script
- **H3_compounding_harm_comparison.csv** - Comparison table of baseline FAR vs adversarial ASR disparity ratios
- **H3_compounding_harm_summary.txt** - Text summary of findings
- **H3_Compounding_Harm.png** - Side-by-side bar charts comparing baseline and adversarial disparities

## Running the Analysis

```bash
cd hypothesis_testing/H3_Compounding_Harm
python H3_Compounding_Harm.py
```

## Results

The analysis compares:
- **Baseline FAR Disparity Ratio**: Max FAR / Min FAR across races
- **Adversarial ASR Disparity Ratio**: Max ASR / Min ASR across races

If the adversarial disparity ratio is greater than the baseline ratio, it indicates that attacks compound existing harm. If it's smaller, attacks may actually reduce disparities.

## Interpretation

- **Harm Compounded**: Adversarial attacks increase existing demographic disparities
- **Harm Not Compounded**: Adversarial attacks do not increase (or may reduce) existing disparities

