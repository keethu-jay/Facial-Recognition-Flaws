# Adversarial Robustness Analysis Report

## Methodology

This study evaluates the demographic fairness of FaceNet under adversarial attacks using controlled negative pairing and white-box adversarial attacks.

### Target Model

- **Model**: FaceNet (Inception ResNet v1)
- **Source**: keras-facenet library (open-source)
- **Pre-trained weights**: VGGFace2 dataset
- **Embedding dimension**: 512
- **Verification threshold (τ)**: 1.0

See `MODEL_DOCUMENTATION.md` for full model attribution and references.

### Dataset

- **Source**: FairFace validation split
- **Demographics**: 7 races × 2 genders × 9 age ranges
- **Images per race**: ~90 (5 male + 5 female per age range)
- **Total images**: 621

### Controlled Negative Pairing

For baseline evaluation, we generate negative pairs (different people) within the same demographic group:
- Same race
- Same gender  
- Same age range

This ensures fair comparison across demographic groups.

### Adversarial Attacks

We test three white-box attack methods:

1. **FGSM** (Fast Gradient Sign Method): Single-step gradient-based attack
2. **PGD** (Projected Gradient Descent): Iterative attack with projection (10 steps)
3. **C&W** (Carlini & Wagner): Optimization-based attack

Each attack is tested with three perturbation budgets:
- ε = 0.01 (subtle)
- ε = 0.03 (moderate)
- ε = 0.05 (more noticeable)

**Total datasets**: 9 (3 attacks × 3 epsilons)

### Attack Strategy: Impersonation

For each baseline pair (Image A, Image B), we:
1. Perturb Image A to minimize distance to Image B's embedding
2. Test if the perturbed Image A is misidentified as Image B (distance < 1.0)
3. Record Attack Success Rate (ASR) = successful attacks / total pairs

---

## Results

### H1: Demographic Disparity Hypothesis

**Question**: Are there significant differences in Attack Success Rate (ASR) across demographic groups?

**Analysis**: Three-Way ANOVA on ASR for Race, Gender, and Age factors.

**Results**: See `outputs/2_Hypothesis_Stats/H1_ANOVA_PGD_e0.03.csv`

**Visualization**: ![ASR Bar Chart](outputs/3_Final_Visuals/H1_ASR_Bar_Chart.png)

**Key Findings**:
- [ANOVA results will be populated after running H1_ASR_ANOVA.py]
- Significant main effects and interactions are identified
- p-values < 0.05 indicate significant demographic disparities

---

### H2: Perturbation Budget Sensitivity

**Question**: Do demographic disparities change with perturbation magnitude?

**Analysis**: Compare most vulnerable vs least vulnerable racial groups across epsilon values.

**Results**: See `outputs/2_Hypothesis_Stats/H2_epsilon_sensitivity_data.csv`

**Visualization**: ![Epsilon Sensitivity](outputs/3_Final_Visuals/H2_Epsilon_Sensitivity.png)

**Key Findings**:
- [Results will show if disparity widens or narrows with increasing epsilon]
- Most vulnerable race: [Identified from baseline]
- Least vulnerable race: [Identified from baseline]

---

### H3: Compounding Harm Analysis

**Question**: Do adversarial attacks compound existing demographic disparities?

**Analysis**: Compare baseline FAR disparity ratio to adversarial ASR disparity ratio.

**Results**: See `outputs/2_Hypothesis_Stats/H3_compounding_harm_comparison.csv`

**Summary**: See `outputs/2_Hypothesis_Stats/H3_compounding_harm_summary.txt`

**Visualization**: ![Compounding Harm](outputs/3_Final_Visuals/H3_Compounding_Harm.png)

**Key Findings**:

#### Baseline FAR Disparity
- **Ratio**: [X]:1 (Max/Min)
- Most Vulnerable Race: [Race] (FAR = [value])
- Least Vulnerable Race: [Race] (FAR = [value])

#### Adversarial ASR Disparity (PGD, ε=0.03)
- **Ratio**: [Y]:1 (Max/Min)
- Most Vulnerable Race: [Race] (ASR = [value])
- Least Vulnerable Race: [Race] (ASR = [value])

#### Conclusion
- [Harm IS/NOT compounded based on comparison]
- Baseline disparity: [X]:1
- Adversarial disparity: [Y]:1
- Change: [±Z]%

**Note**: The **5:1 baseline ratio** mentioned in the study refers to the ratio between the most and least vulnerable racial groups in baseline FAR.

---

### H4: Attack Method Consistency

**Question**: Do different attack methods introduce consistent or different levels of demographic disparity?

**Analysis**: Compare ASR Disparity Ratio across FGSM, PGD, and C&W attacks (ε=0.03).

**Results**: See `outputs/2_Hypothesis_Stats/H4_attack_consistency.csv`

**Visualization**: ![Attack Consistency](outputs/3_Final_Visuals/H4_Attack_Consistency.png)

**Key Findings**:
- FGSM Disparity Ratio: [X]:1
- PGD Disparity Ratio: [Y]:1
- C&W Disparity Ratio: [Z]:1
- Highest disparity method: [Attack Type]

---

## Final Visualizations

### Heatmaps

1. **Baseline FAR Heatmap**: ![Baseline Heatmap](outputs/3_Final_Visuals/Baseline_FAR_Heatmap.png)
   - Shows False Acceptance Rate across Race (rows) and Age Group (columns)
   - Red = High vulnerability, Blue = Low vulnerability

2. **Adversarial ASR Heatmap**: ![Adversarial Heatmap](outputs/3_Final_Visuals/Adversarial_ASR_Heatmap.png)
   - Shows Attack Success Rate across Race (rows) and Age Group (columns)
   - PGD attack, ε=0.03
   - Red = High vulnerability, Blue = Low vulnerability

### Additional Visualizations

- H1 ASR Bar Chart: `outputs/3_Final_Visuals/H1_ASR_Bar_Chart.png`
- H2 Epsilon Sensitivity: `outputs/3_Final_Visuals/H2_Epsilon_Sensitivity.png`
- H3 Compounding Harm: `outputs/3_Final_Visuals/H3_Compounding_Harm.png`
- H4 Attack Consistency: `outputs/3_Final_Visuals/H4_Attack_Consistency.png`

---

## Statistical Outputs

All statistical results are saved in `outputs/2_Hypothesis_Stats/`:

- `H1_ANOVA_PGD_e0.03.csv` - ANOVA results
- `H1_aggregated_ASR_PGD_e0.03.csv` - Aggregated ASR by demographics
- `H2_epsilon_sensitivity_data.csv` - Epsilon sensitivity data
- `H3_compounding_harm_comparison.csv` - Baseline vs Adversarial comparison
- `H3_compounding_harm_summary.txt` - Text summary of H3 results
- `H4_attack_consistency.csv` - Attack method comparison

---

## Reproducibility

All parameters are defined in `config.py`:
- Distance threshold: 1.0
- Attack types: FGSM, PGD, CW
- Epsilon values: 0.01, 0.03, 0.05
- PGD steps: 10
- Significance level: 0.05

---

## References

1. FaceNet Model: keras-facenet library - https://github.com/SergeyDmitriev/keras-facenet
2. Original FaceNet Paper: Schroff et al., "FaceNet: A unified embedding for face recognition and clustering" (2015)
3. Attack Methods:
   - FGSM: Goodfellow et al., "Explaining and Harnessing Adversarial Examples" (2014)
   - PGD: Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (2017)
   - C&W: Carlini & Wagner, "Towards Evaluating the Robustness of Neural Networks" (2017)

---

*Report generated automatically. Run all hypothesis scripts to populate results.*

