"""
Final Visualization Generation

Creates heatmaps comparing baseline FAR to adversarial ASR
across Race and Age dimensions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    BASELINE_DIR, ATTACK_TYPES, EPSILONS, RACES, AGE_GROUPS, COLOR_PALETTE,
    get_adversarial_data_path
)


def load_baseline_fsr():
    """Load baseline FSR and aggregate by race and age"""
    baseline_path = os.path.join(BASELINE_DIR, 'all_races_baseline.csv')
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline data not found: {baseline_path}")
    
    df = pd.read_csv(baseline_path)
    
    # Aggregate by race and age
    race_age_fsr = df.groupby(['race', 'age'])['fsr'].mean().reset_index()
    
    # Pivot for heatmap
    heatmap_data = race_age_fsr.pivot(index='race', columns='age', values='fsr')
    
    # Reorder rows and columns
    heatmap_data = heatmap_data.reindex(index=RACES)
    heatmap_data = heatmap_data.reindex(columns=AGE_GROUPS)
    
    return heatmap_data


def load_adversarial_asr(attack_type, epsilon):
    """Load adversarial ASR and aggregate by race and age"""
    filepath = get_adversarial_data_path(attack_type, epsilon)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Adversarial data not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Aggregate by race and age
    race_age_asr = df.groupby(['race', 'age'])['Attack_Success_Status'].mean().reset_index()
    
    # Pivot for heatmap
    heatmap_data = race_age_asr.pivot(index='race', columns='age', values='Attack_Success_Status')
    
    # Reorder rows and columns
    heatmap_data = heatmap_data.reindex(index=RACES)
    heatmap_data = heatmap_data.reindex(columns=AGE_GROUPS)
    
    return heatmap_data


def create_heatmap(data, title, filename, vmin=None, vmax=None):
    """Create a heatmap visualization"""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Create heatmap
    sns.heatmap(data, annot=True, fmt='.3f', cmap=COLOR_PALETTE,
               vmin=vmin, vmax=vmax, center=None,
               cbar_kws={'label': 'Rate'},
               linewidths=0.5, linecolor='gray',
               ax=ax)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax.set_ylabel('Race', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {filename}")


def main():
    """Main execution"""
    print("\n" + "*" * 70)
    print("FINAL VISUALIZATION GENERATION")
    print("*" * 70)
    
    # Create output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Baseline Heatmap
    print("\n1. Creating baseline FAR heatmap...")
    try:
        baseline_heatmap = load_baseline_fsr()
        create_heatmap(
            baseline_heatmap,
            'Baseline False Acceptance Rate (FAR)\nby Race and Age Group',
            'Baseline_FAR_Heatmap.png'
        )
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    # 2. Adversarial Heatmap (PGD, epsilon=0.03)
    print("\n2. Creating adversarial ASR heatmap...")
    attack_type = 'PGD'
    epsilon = 0.03
    
    try:
        adversarial_heatmap = load_adversarial_asr(attack_type, epsilon)
        create_heatmap(
            adversarial_heatmap,
            f'Adversarial Attack Success Rate (ASR)\nby Race and Age Group\n{attack_type} Attack, ε={epsilon}',
            'Adversarial_ASR_Heatmap.png'
        )
    except FileNotFoundError:
        print(f"  ✗ Adversarial data not found. Trying other epsilon values...")
        for eps in EPSILONS:
            try:
                adversarial_heatmap = load_adversarial_asr(attack_type, eps)
                epsilon = eps
                create_heatmap(
                    adversarial_heatmap,
                    f'Adversarial Attack Success Rate (ASR)\nby Race and Age Group\n{attack_type} Attack, ε={epsilon}',
                    'Adversarial_ASR_Heatmap.png'
                )
                break
            except FileNotFoundError:
                continue
        else:
            print("  ✗ No adversarial data found. Please run generate_adv_data.py first.")
    except Exception as e:
        print(f"  ✗ Error: {e}")
    
    print("\n" + "*" * 70)
    print("FINAL VISUALIZATIONS COMPLETE!")
    print("*" * 70)
    output_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

