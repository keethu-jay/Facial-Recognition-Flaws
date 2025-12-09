# H4: Attack Method Consistency Analysis

This folder contains the analysis for testing whether different attack methods (FGSM, PGD, C&W) introduce consistent or different levels of demographic disparity.

## Files

- **H4_Attack_Consistency.py** - Main analysis script
- **H4_attack_consistency.csv** - Disparity ratios for each attack method
- **H4_Attack_Consistency.png** - Bar chart comparing disparity ratios across attack types

## Running the Analysis

```bash
cd hypothesis_testing/H4_Attack_Consistency
python H4_Attack_Consistency.py
```

## Results

The analysis calculates the ASR Disparity Ratio (Max ASR / Min ASR) for each attack method at epsilon=0.03:
- **FGSM**: Fast Gradient Sign Method
- **PGD**: Projected Gradient Descent
- **CW**: Carlini & Wagner

## Interpretation

- **Consistent Disparities**: All attacks show similar disparity ratios → demographic bias is attack-independent
- **Inconsistent Disparities**: Different attacks show different ratios → some attacks may be more/less fair

This helps determine if demographic disparities are inherent to the model or specific to certain attack methods.

