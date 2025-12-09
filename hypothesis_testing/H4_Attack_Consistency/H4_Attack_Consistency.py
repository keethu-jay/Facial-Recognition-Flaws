"""
H4: Attack Method Consistency Analysis

Tests whether different attack methods (FGSM, PGD, C&W) introduce
consistent or different levels of demographic disparity.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    ATTACK_TYPES, EPSILONS, RACES, COLOR_PALETTE,
    get_adversarial_data_path
)


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
    """Calculate disparity ratio: max / min"""
    max_val = values.max()
    min_val = values.min()
    
    ratio = max_val / min_val if min_val > 0 else float('inf')
    
    return ratio


def main():
    """Main execution"""
    print("\n" + "*" * 70)
    print("H4: ATTACK METHOD CONSISTENCY ANALYSIS")
    print("*" * 70)
    
    # Create output directories
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Use consistent epsilon=0.03 as specified
    epsilon = 0.03
    
    print(f"\nAnalyzing attack methods with eps={epsilon}")
    
    # Calculate disparity ratio for each attack type
    disparity_data = []
    
    for attack_type in ATTACK_TYPES:
        try:
            print(f"\nProcessing {attack_type}...")
            race_asr = load_adversarial_asr(attack_type, epsilon)
            
            disparity_ratio = calculate_disparity_ratio(race_asr)
            max_race = race_asr.idxmax()
            min_race = race_asr.idxmin()
            max_asr = race_asr.max()
            min_asr = race_asr.min()
            
            disparity_data.append({
                'attack_type': attack_type,
                'disparity_ratio': disparity_ratio,
                'max_race': max_race,
                'max_asr': max_asr,
                'min_race': min_race,
                'min_asr': min_asr
            })
            
            print(f"  Disparity Ratio: {disparity_ratio:.2f}:1")
            print(f"    Max: {max_race} (ASR={max_asr:.4f})")
            print(f"    Min: {min_race} (ASR={min_asr:.4f})")
            
        except FileNotFoundError:
            print(f"  [ERROR] Data not found for {attack_type}")
            continue
    
    if len(disparity_data) == 0:
        print("\n[ERROR] No adversarial data found. Please run generate_adv_data.py first.")
        return
    
    # Create DataFrame
    disparity_df = pd.DataFrame(disparity_data)
    disparity_df = disparity_df.sort_values('disparity_ratio', ascending=False)
    
    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(output_dir, 'H4_attack_consistency.csv')
    disparity_df.to_csv(results_path, index=False)
    print(f"\n  [OK] Saved results: {results_path}")
    
    # Create bar chart
    print("\nCreating disparity ratio comparison chart...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Use color map
    cmap = plt.cm.get_cmap(COLOR_PALETTE)
    colors = [cmap(i / len(disparity_df)) for i in range(len(disparity_df))]
    
    bars = ax.bar(disparity_df['attack_type'], disparity_df['disparity_ratio'],
                 color=colors, alpha=0.8, edgecolor='black', linewidth=2)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}:1',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax.set_xlabel('Attack Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('ASR Disparity Ratio (Max/Min)', fontsize=12, fontweight='bold')
    ax.set_title(f'Attack Method Consistency: Disparity Ratio Comparison\n'
                f'eps={epsilon} (Higher = Greater Demographic Disparity)', 
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, max(disparity_df['disparity_ratio']) * 1.2])
    
    # Add annotation
    max_attack = disparity_df.iloc[0]['attack_type']
    max_ratio = disparity_df.iloc[0]['disparity_ratio']
    ax.text(0.5, 0.95, f'Highest disparity: {max_attack} ({max_ratio:.2f}:1)',
           transform=ax.transAxes, ha='center', fontsize=10,
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, 'H4_Attack_Consistency.png'), 
               dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Saved: H4_Attack_Consistency.png")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ATTACK CONSISTENCY SUMMARY")
    print("=" * 70)
    print(disparity_df.to_string(index=False))
    
    print("\n" + "*" * 70)
    print("H4 ANALYSIS COMPLETE!")
    print("*" * 70)


if __name__ == "__main__":
    main()

