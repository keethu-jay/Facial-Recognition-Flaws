"""
H3: Compounding Harm Analysis

Compares baseline FAR disparity to adversarial ASR disparity to determine
if adversarial attacks compound existing demographic disparities.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from config import (
    BASELINE_DIR, ATTACK_TYPES, EPSILONS, RACES,
    get_adversarial_data_path
)


def load_baseline_fsr():
    """Load baseline FSR (FAR) by race"""
    baseline_path = os.path.join(BASELINE_DIR, 'all_races_baseline.csv')
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline data not found: {baseline_path}")
    
    df = pd.read_csv(baseline_path)
    
    # Calculate mean FSR (FAR) by race
    race_fsr = df.groupby('race')['fsr'].mean()
    
    return race_fsr


def load_adversarial_asr(attack_type, epsilon):
    """Load adversarial ASR by race"""
    filepath = get_adversarial_data_path(attack_type, epsilon)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Adversarial data not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Calculate mean ASR by race
    race_asr = df.groupby('race')['Attack_Success_Status'].mean()
    
    return race_asr


def calculate_disparity_ratio(values):
    """
    Calculate disparity ratio: max / min
    
    Returns ratio and identifies which groups are max/min
    """
    max_val = values.max()
    min_val = values.min()
    max_race = values.idxmax()
    min_race = values.idxmin()
    
    ratio = max_val / min_val if min_val > 0 else float('inf')
    
    return ratio, max_race, min_race, max_val, min_val


def main():
    """Main execution"""
    print("\n" + "*" * 70)
    print("H3: COMPOUNDING HARM ANALYSIS")
    print("*" * 70)
    
    # Create output directories
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load baseline FAR
    print("\nLoading baseline FAR data...")
    baseline_fsr = load_baseline_fsr()
    print(f"  [OK] Loaded baseline FSR for {len(baseline_fsr)} races")
    
    # Calculate baseline disparity ratio
    print("\nCalculating baseline disparity ratio...")
    far_ratio, far_max_race, far_min_race, far_max_val, far_min_val = calculate_disparity_ratio(baseline_fsr)
    print(f"  Baseline FAR Ratio: {far_ratio:.2f}:1")
    print(f"    Max: {far_max_race} (FAR={far_max_val:.4f})")
    print(f"    Min: {far_min_race} (FAR={far_min_val:.4f})")
    
    # Use PGD, epsilon=0.03 (or highest disparity)
    attack_type = 'PGD'
    epsilon = 0.03
    
    print(f"\nLoading adversarial ASR data ({attack_type}, eps={epsilon})...")
    try:
        adversarial_asr = load_adversarial_asr(attack_type, epsilon)
        print(f"  ✓ Loaded adversarial ASR for {len(adversarial_asr)} races")
    except FileNotFoundError:
        print(f"  [WARNING] Adversarial data not found. Trying other epsilon values...")
        # Try other epsilons
        for eps in EPSILONS:
            try:
                adversarial_asr = load_adversarial_asr(attack_type, eps)
                epsilon = eps
                print(f"  [OK] Using eps={epsilon} instead")
                break
            except FileNotFoundError:
                continue
        else:
            print("  [ERROR] No adversarial data found. Please run generate_adv_data.py first.")
            return
    
    # Calculate adversarial disparity ratio
    print("\nCalculating adversarial disparity ratio...")
    asr_ratio, asr_max_race, asr_min_race, asr_max_val, asr_min_val = calculate_disparity_ratio(adversarial_asr)
    print(f"  Adversarial ASR Ratio: {asr_ratio:.2f}:1")
    print(f"    Max: {asr_max_race} (ASR={asr_max_val:.4f})")
    print(f"    Min: {asr_min_race} (ASR={asr_min_val:.4f})")
    
    # Create comparison table
    comparison_data = {
        'Metric': ['Baseline FAR Ratio', 'Adversarial ASR Ratio'],
        'Ratio': [far_ratio, asr_ratio],
        'Max_Race': [far_max_race, asr_max_race],
        'Max_Value': [far_max_val, asr_max_val],
        'Min_Race': [far_min_race, asr_min_race],
        'Min_Value': [far_min_val, asr_min_val]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(output_dir, 'H3_compounding_harm_comparison.csv')
    comparison_df.to_csv(results_path, index=False)
    print(f"\n  ✓ Saved comparison table: {results_path}")
    
    # Determine if harm is compounded
    harm_compounded = asr_ratio > far_ratio
    
    # Create summary text
    summary_text = f"""
COMPOUNDING HARM ANALYSIS RESULTS
==================================

Baseline FAR Disparity:
  Ratio: {far_ratio:.2f}:1
  Most Vulnerable: {far_max_race} (FAR = {far_max_val:.4f})
  Least Vulnerable: {far_min_race} (FAR = {far_min_val:.4f})

Adversarial ASR Disparity ({attack_type}, eps={epsilon}):
  Ratio: {asr_ratio:.2f}:1
  Most Vulnerable: {asr_max_race} (ASR = {asr_max_val:.4f})
  Least Vulnerable: {asr_min_race} (ASR = {asr_min_val:.4f})

CONCLUSION:
{'[YES] Harm IS COMPOUNDED' if harm_compounded else '[NO] Harm is NOT compounded'}
  The adversarial attack {'increases' if harm_compounded else 'does not increase'} 
  the existing demographic disparity.
  
  Baseline disparity: {far_ratio:.2f}:1
  Adversarial disparity: {asr_ratio:.2f}:1
  Change: {((asr_ratio / far_ratio - 1) * 100):+.1f}%
"""
    
    print(summary_text)
    
    # Save summary
    output_dir = os.path.dirname(os.path.abspath(__file__))
    summary_path = os.path.join(output_dir, 'H3_compounding_harm_summary.txt')
    with open(summary_path, 'w') as f:
        f.write(summary_text)
    print(f"  [OK] Saved summary: {summary_path}")
    
    # Create visualization
    print("\nCreating comparison visualization...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Baseline FAR
    baseline_fsr_sorted = baseline_fsr.sort_values(ascending=False)
    colors1 = plt.cm.coolwarm_r(np.linspace(0.2, 0.8, len(baseline_fsr_sorted)))
    bars1 = ax1.bar(baseline_fsr_sorted.index, baseline_fsr_sorted.values, 
                   color=colors1, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_title('Baseline FAR by Race', fontsize=12, fontweight='bold')
    ax1.set_ylabel('False Acceptance Rate (FAR)', fontsize=11)
    ax1.set_xticklabels(baseline_fsr_sorted.index, rotation=45, ha='right')
    ax1.grid(axis='y', alpha=0.3)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    # Adversarial ASR
    adversarial_asr_sorted = adversarial_asr.sort_values(ascending=False)
    colors2 = plt.cm.coolwarm_r(np.linspace(0.2, 0.8, len(adversarial_asr_sorted)))
    bars2 = ax2.bar(adversarial_asr_sorted.index, adversarial_asr_sorted.values,
                   color=colors2, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_title(f'Adversarial ASR by Race\n{attack_type} Attack, eps={epsilon}', 
                 fontsize=12, fontweight='bold')
    ax2.set_ylabel('Attack Success Rate (ASR)', fontsize=11)
    ax2.set_xticklabels(adversarial_asr_sorted.index, rotation=45, ha='right')
    ax2.grid(axis='y', alpha=0.3)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Compounding Harm Analysis: Baseline FAR vs Adversarial ASR', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'H3_Compounding_Harm.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: H3_Compounding_Harm.png")
    
    print("\n" + "*" * 70)
    print("H3 ANALYSIS COMPLETE!")
    print("*" * 70)


if __name__ == "__main__":
    main()

