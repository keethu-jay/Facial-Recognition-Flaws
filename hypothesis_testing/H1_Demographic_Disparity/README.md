# H1: Demographic Disparity Analysis

This folder contains the analysis for testing whether there are significant differences in Attack Success Rate (ASR) across demographic groups.

## Files

- **H1_ASR_ANOVA.py** - Main analysis script that performs Three-Way ANOVA (Race × Gender × Age)
- **H1_aggregated_ASR_PGD_e0.03.csv** - Aggregated ASR statistics by demographic subgroups
- **H1_ANOVA_PGD_e0.03.csv** - ANOVA results table with p-values and F-statistics
- **H1_ASR_Bar_Chart.png** - Visualization showing mean ASR by race

## Running the Analysis

```bash
cd hypothesis_testing/H1_Demographic_Disparity
python H1_ASR_ANOVA.py
```

## Results

The analysis uses PGD attack with epsilon=0.03 as the primary test. It tests for significant effects of:
- Race
- Gender  
- Age
- All two-way and three-way interactions

Significant effects (p < 0.05) indicate demographic disparities in adversarial vulnerability.

