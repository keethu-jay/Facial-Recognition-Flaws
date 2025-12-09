"""
H2: Perturbation Budget Sensitivity Analysis

Tests whether demographic disparities change with perturbation magnitude (epsilon).
Compares most vulnerable vs least vulnerable racial groups across epsilon values.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    BASELINE_DIR, ATTACK_TYPES, EPSILONS, RACES, COLOR_PALETTE,
    get_adversarial_data_path
)


def load_baseline_fsr():
    """Load baseline FSR to identify most/least vulnerable races"""
    baseline_path = os.path.join(BASELINE_DIR, 'all_races_baseline.csv')
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline data not found: {baseline_path}")
    
    df = pd.read_csv(baseline_path)
    
    # Calculate mean FSR by race
    race_fsr = df.groupby('race')['fsr'].mean().sort_values(ascending=False)
    
    return df, race_fsr


def identify_extreme_races(race_fsr):
    """
    Identify most vulnerable (highest FSR) and least vulnerable (lowest FSR) races
    """
    most_vulnerable = race_fsr.index[0]  # Highest FSR
    least_vulnerable = race_fsr.index[-1]  # Lowest FSR
    
    print(f"  Most Vulnerable Race: {most_vulnerable} (FSR: {race_fsr[most_vulnerable]:.4f})")
    print(f"  Least Vulnerable Race: {least_vulnerable} (FSR: {race_fsr[least_vulnerable]:.4f})")
    
    return most_vulnerable, least_vulnerable


def load_adversarial_data(attack_type, epsilon):
    """Load adversarial data for specific attack and epsilon"""
    filepath = get_adversarial_data_path(attack_type, epsilon)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Adversarial data not found: {filepath}")
    
    return pd.read_csv(filepath)


def calculate_asr_by_race(df):
    """Calculate mean ASR for each race"""
    race_asr = df.groupby('race')['Attack_Success_Status'].mean()
    return race_asr


def create_epsilon_sensitivity_plot(most_vuln_race, least_vuln_race, attack_type, output_dir):
    """
    Create line plot showing ASR vs Epsilon for extreme races
    """
    # Collect data for all epsilons
    epsilons_list = []
    most_vuln_asr = []
    least_vuln_asr = []
    
    for epsilon in EPSILONS:
        try:
            df = load_adversarial_data(attack_type, epsilon)
            race_asr = calculate_asr_by_race(df)
            
            epsilons_list.append(epsilon)
            most_vuln_asr.append(race_asr.get(most_vuln_race, 0))
            least_vuln_asr.append(race_asr.get(least_vuln_race, 0))
            
        except FileNotFoundError:
            print(f"  Warning: Data not found for {attack_type} eps={epsilon}")
            continue
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot lines
    ax.plot(epsilons_list, most_vuln_asr, marker='o', linewidth=3, markersize=10,
           label=f'Most Vulnerable: {most_vuln_race}', color='#e74c3c')
    ax.plot(epsilons_list, least_vuln_asr, marker='s', linewidth=3, markersize=10,
           label=f'Least Vulnerable: {least_vuln_race}', color='#3498db')
    
    # Add value labels
    for i, (eps, asr) in enumerate(zip(epsilons_list, most_vuln_asr)):
        ax.text(eps, asr, f'{asr:.2%}', ha='center', va='bottom', fontweight='bold')
    for i, (eps, asr) in enumerate(zip(epsilons_list, least_vuln_asr)):
        ax.text(eps, asr, f'{asr:.2%}', ha='center', va='top', fontweight='bold')
    
    ax.set_xlabel('Perturbation Budget (eps)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Success Rate (ASR)', fontsize=12, fontweight='bold')
    ax.set_title(f'Epsilon Sensitivity Analysis\n'
                f'ASR vs. Perturbation Budget for Extreme Racial Groups\n'
                f'{attack_type} Attack', 
                fontsize=14, fontweight='bold')
    ax.set_xticks(EPSILONS)
    ax.set_xticklabels([f'{e:.2f}' for e in EPSILONS])
    ax.legend(fontsize=11, loc='best')
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_ylim([0, max(max(most_vuln_asr), max(least_vuln_asr)) * 1.2])
    
    # Add annotation about disparity
    if len(epsilons_list) > 0:
        disparity_start = most_vuln_asr[0] - least_vuln_asr[0]
        disparity_end = most_vuln_asr[-1] - least_vuln_asr[-1]
        
        if disparity_end > disparity_start:
            trend = "widens"
        elif disparity_end < disparity_start:
            trend = "narrows"
        else:
            trend = "remains constant"
        
        ax.text(0.5, 0.02, f'Disparity {trend} with increasing eps',
               transform=ax.transAxes, ha='center', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'H2_Epsilon_Sensitivity.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: H2_Epsilon_Sensitivity.png")


def main():
    """Main execution"""
    print("\n" + "*" * 70)
    print("H2: PERTURBATION BUDGET SENSITIVITY ANALYSIS")
    print("*" * 70)
    
    # Create output directories
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load baseline to identify extreme races
    print("\nLoading baseline data...")
    baseline_df, race_fsr = load_baseline_fsr()
    
    # Identify most and least vulnerable races
    print("\nIdentifying extreme races...")
    most_vuln, least_vuln = identify_extreme_races(race_fsr)
    
    # Use strongest attack (PGD) as specified
    attack_type = 'PGD'
    print(f"\nUsing {attack_type} attack for analysis")
    
    # Collect ASR data across all epsilons
    print("\nCollecting ASR data across epsilon values...")
    epsilon_data = []
    
    for epsilon in EPSILONS:
        try:
            df = load_adversarial_data(attack_type, epsilon)
            race_asr = calculate_asr_by_race(df)
            
            epsilon_data.append({
                'epsilon': epsilon,
                'most_vulnerable_race': most_vuln,
                'most_vulnerable_asr': race_asr.get(most_vuln, 0),
                'least_vulnerable_race': least_vuln,
                'least_vulnerable_asr': race_asr.get(least_vuln, 0),
                'disparity': race_asr.get(most_vuln, 0) - race_asr.get(least_vuln, 0)
            })
            
            print(f"  eps={epsilon:.2f}: {most_vuln} ASR={race_asr.get(most_vuln, 0):.2%}, "
                  f"{least_vuln} ASR={race_asr.get(least_vuln, 0):.2%}")
            
        except FileNotFoundError:
            print(f"  [ERROR] Data not found for eps={epsilon}")
            continue
    
    # Save data
    epsilon_df = pd.DataFrame(epsilon_data)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    epsilon_path = os.path.join(output_dir, 'H2_epsilon_sensitivity_data.csv')
    epsilon_df.to_csv(epsilon_path, index=False)
    print(f"\n  [OK] Saved: {epsilon_path}")
    
    # Create visualization
    print("\nCreating epsilon sensitivity plot...")
    output_dir = os.path.dirname(os.path.abspath(__file__))
    create_epsilon_sensitivity_plot(most_vuln, least_vuln, attack_type, output_dir)
    
    # Print summary
    print("\n" + "=" * 70)
    print("EPSILON SENSITIVITY SUMMARY")
    print("=" * 70)
    print(epsilon_df.to_string(index=False))
    
    print("\n" + "*" * 70)
    print("H2 ANALYSIS COMPLETE!")
    print("*" * 70)


if __name__ == "__main__":
    main()

