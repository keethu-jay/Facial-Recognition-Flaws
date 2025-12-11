# Project Structure

This document describes the reorganized project structure.

## Directory Organization

```
Facial-Recognition-Flaws/
├── datasets/                          # All dataset-related files
│   ├── baseline_stats/                # Baseline statistics and pairs
│   │   ├── all_races_baseline.csv
│   │   ├── all_races_pairs.csv
│   │   ├── *_baseline.csv             # Per-race baseline stats
│   │   ├── *_pairs.csv                # Per-race pairs
│   │   └── *.png                      # Baseline visualizations
│   ├── results/                       # Adversarial attack results
│   │   ├── FGSM/                      # FGSM attack results
│   │   │   ├── e0.01/
│   │   │   │   └── FGSM_e0.01_data.csv
│   │   │   ├── e0.03/
│   │   │   │   └── FGSM_e0.03_data.csv
│   │   │   └── e0.05/
│   │   │       └── FGSM_e0.05_data.csv
│   │   ├── PGD/                       # PGD attack results
│   │   │   ├── e0.01/
│   │   │   ├── e0.03/
│   │   │   └── e0.05/
│   │   └── CW/                        # C&W attack results
│   │       ├── e0.01/
│   │       ├── e0.03/
│   │       └── e0.05/
│   ├── Black.pkl, White.pkl, etc.     # Original race datasets
│   └── Black/, White/, etc./          # Race-specific image directories
│
├── hypothesis_testing/                 # All hypothesis analysis
│   ├── H1_Demographic_Disparity/      # H1: ANOVA analysis
│   │   ├── H1_ASR_ANOVA.py
│   │   ├── H1_aggregated_ASR_*.csv
│   │   ├── H1_ANOVA_*.csv
│   │   ├── H1_ASR_Bar_Chart.png
│   │   └── README.md
│   ├── H2_Epsilon_Sensitivity/        # H2: Epsilon sensitivity
│   │   ├── H2_Epsilon_Sensitivity.py
│   │   ├── H2_epsilon_sensitivity_data.csv
│   │   ├── H2_Epsilon_Sensitivity.png
│   │   └── README.md
│   ├── H3_Compounding_Harm/           # H3: Compounding harm
│   │   ├── H3_Compounding_Harm.py
│   │   ├── H3_compounding_harm_*.csv
│   │   ├── H3_compounding_harm_summary.txt
│   │   ├── H3_Compounding_Harm.png
│   │   └── README.md
│   ├── H4_Attack_Consistency/         # H4: Attack consistency
│   │   ├── H4_Attack_Consistency.py
│   │   ├── H4_attack_consistency.csv
│   │   ├── H4_Attack_Consistency.png
│   │   └── README.md
│   ├── run_all_hypothesis_tests.py    # Master script to run all tests
│   ├── create_final_visualizations.py # Creates heatmaps
│   ├── analysis_report.md             # Combined analysis report
│   ├── Baseline_FAR_Heatmap.png       # Baseline heatmap
│   └── Adversarial_ASR_Heatmap.png    # Adversarial heatmap
│
├── Datasets/                          # Dataset creation scripts
│   ├── config.py                      # Configuration file
│   ├── DatasetCreation.py             # Creates original datasets
│   ├── CreateBaselineStats.py        # Creates baseline statistics
│   ├── CreatePerturbedDatasets.py    # Creates perturbed datasets
│   └── convert_consolidated_to_individual.py  # Utility script
│
└── [other project files]
```

## Key Changes

1. **Datasets organized by type**: Baseline stats and attack results are clearly separated
2. **Attack results by method**: Each attack (FGSM, PGD, CW) has its own folder with epsilon subfolders
3. **Hypothesis testing organized**: Each hypothesis has its own folder with scripts, results, and visualizations
4. **Removed unnecessary files**: Deleted Colab scripts, debugging files, and test scripts

## Running Analyses

### Run All Hypothesis Tests
```bash
cd hypothesis_testing
python run_all_hypothesis_tests.py
```

### Run Individual Hypothesis
```bash
cd hypothesis_testing/H1_Demographic_Disparity
python H1_ASR_ANOVA.py
```

### Create Final Visualizations
```bash
cd hypothesis_testing
python create_final_visualizations.py
```

## Data Flow

1. **Original Datasets**: Created by `DatasetCreation.py` → stored in `datasets/`
2. **Baseline Statistics**: Created by `CreateBaselineStats.py` → stored in `datasets/baseline_stats/`
3. **Adversarial Results**: Created from attacks → stored in `datasets/results/{ATTACK}/{EPSILON}/`
4. **Hypothesis Analysis**: Scripts read from `datasets/` and save results to their respective folders




