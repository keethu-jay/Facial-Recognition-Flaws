"""
H1: Demographic Disparity Hypothesis Analysis

Tests whether there are significant differences in Attack Success Rate (ASR)
across demographic groups (Race, Gender, Age) using Three-Way ANOVA.
"""

import os
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    HYPOTHESIS_STATS_DIR, FINAL_VISUALS_DIR,
    ATTACK_TYPES, EPSILONS, RACES, COLOR_PALETTE, SIGNIFICANCE_ALPHA,
    get_adversarial_data_path
)


def load_adversarial_data(attack_type, epsilon):
    """Load adversarial data for specific attack and epsilon"""
    filepath = get_adversarial_data_path(attack_type, epsilon)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Adversarial data not found: {filepath}")
    
    df = pd.read_csv(filepath)
    return df


def aggregate_asr_by_demographics(df):
    """
    Aggregate ASR by Race × Gender × Age subgroups
    
    Returns DataFrame with mean ASR for each subgroup
    """
    # Group by demographics and calculate mean ASR
    aggregated = df.groupby(['race', 'gender', 'age'])['Attack_Success_Status'].agg([
        'mean',  # Mean ASR
        'count',  # Number of pairs
        'std'     # Standard deviation
    ]).reset_index()
    
    aggregated.columns = ['race', 'gender', 'age', 'mean_asr', 'n_pairs', 'std_asr']
    
    return aggregated


def run_three_way_anova(df):
    """
    Run Three-Way ANOVA on ASR
    
    Tests main effects: Race, Gender, Age
    Tests interactions: Race×Gender, Race×Age, Gender×Age, Race×Gender×Age
    """
    # Prepare data for ANOVA
    # Need to ensure we have enough data points per group
    
    # Create formula for ANOVA
    formula = 'Attack_Success_Status ~ C(race) + C(gender) + C(age) + C(race):C(gender) + C(race):C(age) + C(gender):C(age) + C(race):C(gender):C(age)'
    
    try:
        # Fit the model
        model = ols(formula, data=df).fit()
        
        # Run ANOVA
        anova_results = anova_lm(model, typ=2)
        
        return anova_results, model
    except Exception as e:
        print(f"  Warning: ANOVA failed: {e}")
        print("  Trying simplified model...")
        
        # Try without three-way interaction
        formula_simple = 'Attack_Success_Status ~ C(race) + C(gender) + C(age) + C(race):C(gender) + C(race):C(age) + C(gender):C(age)'
        model = ols(formula_simple, data=df).fit()
        anova_results = anova_lm(model, typ=2)
        
        return anova_results, model


def create_asr_bar_chart(df, attack_type, epsilon, output_dir):
    """Create bar chart showing mean ASR by race"""
    # Aggregate ASR by race
    race_asr = df.groupby('race')['Attack_Success_Status'].mean().sort_values(ascending=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use color map
    cmap = plt.cm.get_cmap(COLOR_PALETTE)
    colors = [cmap(i / len(race_asr)) for i in range(len(race_asr))]
    
    bars = ax.bar(race_asr.index, race_asr.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Race', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Attack Success Rate (ASR)', fontsize=12, fontweight='bold')
    ax.set_title(f'Mean ASR by Race\n{attack_type} Attack, eps={epsilon}', 
                fontsize=14, fontweight='bold')
    ax.set_xticklabels(race_asr.index, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(race_asr.values) * 1.2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'H1_ASR_Bar_Chart.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: H1_ASR_Bar_Chart.png")


def main():
    """Main execution"""
    print("\n" + "*" * 70)
    print("H1: DEMOGRAPHIC DISPARITY HYPOTHESIS (ANOVA)")
    print("*" * 70)
    
    # Create output directories
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Use PGD, epsilon=0.03 as primary test (as specified)
    attack_type = 'PGD'
    epsilon = 0.03
    
    print(f"\nUsing {attack_type} attack with eps={epsilon} for primary analysis")
    
    # Load adversarial data
    print(f"\nLoading adversarial data...")
    df = load_adversarial_data(attack_type, epsilon)
    print(f"  [OK] Loaded {len(df)} adversarial pairs")
    
    # Aggregate ASR by demographics
    print("\nAggregating ASR by demographics...")
    aggregated_df = aggregate_asr_by_demographics(df)
    print(f"  [OK] Created {len(aggregated_df)} demographic subgroups")
    
    # Save aggregated data
    output_dir = os.path.dirname(os.path.abspath(__file__))
    agg_path = os.path.join(output_dir, f'H1_aggregated_ASR_{attack_type}_e{epsilon:.2f}.csv')
    aggregated_df.to_csv(agg_path, index=False)
    print(f"  [OK] Saved aggregated data: {agg_path}")
    
    # Run Three-Way ANOVA
    print("\nRunning Three-Way ANOVA...")
    anova_results, model = run_three_way_anova(df)
    
    # Save ANOVA results
    anova_path = os.path.join(output_dir, f'H1_ANOVA_{attack_type}_e{epsilon:.2f}.csv')
    anova_results.to_csv(anova_path)
    print(f"  [OK] Saved ANOVA results: {anova_path}")
    
    # Print ANOVA summary
    print("\n" + "=" * 70)
    print("ANOVA RESULTS")
    print("=" * 70)
    print(anova_results)
    print("\nSignificant effects (p < 0.05):")
    significant = anova_results[anova_results['PR(>F)'] < SIGNIFICANCE_ALPHA]
    print(significant)
    
    # Create bar chart
    print("\nCreating ASR bar chart...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    create_asr_bar_chart(df, attack_type, epsilon, output_dir)
    
    print("\n" + "*" * 70)
    print("H1 ANALYSIS COMPLETE!")
    print("*" * 70)
    print(f"\nResults saved to:")
    print(f"  - {output_dir}")


if __name__ == "__main__":
    main()

