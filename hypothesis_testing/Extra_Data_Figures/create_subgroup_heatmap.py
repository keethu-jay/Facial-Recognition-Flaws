"""
Create Heatmap of Mean ASR for All 126 Subgroups
(Race × Age × Gender combinations)

This script generates a heatmap showing the mean Attack Success Rate (ASR)
for each of the 126 demographic subgroups from PGD attack with epsilon=0.03.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directories to path to import config
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, 'Datasets'))
from config import RACES, AGE_GROUPS, GENDERS

def load_aggregated_asr_data():
    """Load the aggregated ASR data for PGD attack with epsilon=0.03"""
    # Path to aggregated data
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'H1_Demographic_Disparity',
        'H1_aggregated_ASR_PGD_e0.03.csv'
    )
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    df = pd.read_csv(data_path)
    return df


def create_heatmap_data(df):
    """
    Create a matrix for the heatmap
    
    Rows: Races (7)
    Columns: Age × Gender combinations (9 ages × 2 genders = 18)
    Values: Mean ASR
    """
    # Create column labels: Age_Gender format
    column_labels = []
    for age in AGE_GROUPS:
        for gender in GENDERS:
            column_labels.append(f"{age}\n{gender}")
    
    # Initialize matrix with NaN
    heatmap_matrix = np.full((len(RACES), len(column_labels)), np.nan)
    
    # Fill in the matrix
    for i, race in enumerate(RACES):
        for j, age in enumerate(AGE_GROUPS):
            for k, gender in enumerate(GENDERS):
                col_idx = j * len(GENDERS) + k
                
                # Find matching row in dataframe
                mask = (df['race'] == race) & (df['age'] == age) & (df['gender'] == gender)
                matching_rows = df[mask]
                
                if len(matching_rows) > 0:
                    heatmap_matrix[i, col_idx] = matching_rows.iloc[0]['mean_asr']
    
    return heatmap_matrix, column_labels


def create_heatmap(heatmap_matrix, column_labels, output_path):
    """
    Create and save the heatmap with red-to-green color coding
    
    Red = High ASR (more vulnerable)
    Green = Low ASR (less vulnerable)
    """
    # Create figure with appropriate size
    fig, ax = plt.subplots(figsize=(20, 8))
    
    # Create custom colormap: Red (high) to Green (low)
    # Using RdYlGn (Red-Yellow-Green) reversed so red is high
    cmap = plt.cm.RdYlGn_r
    
    # Create heatmap
    im = ax.imshow(heatmap_matrix, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(column_labels)))
    ax.set_xticklabels(column_labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(np.arange(len(RACES)))
    ax.set_yticklabels(RACES, fontsize=10, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label('Mean ASR', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Add text annotations for each cell
    for i in range(len(RACES)):
        for j in range(len(column_labels)):
            value = heatmap_matrix[i, j]
            if not np.isnan(value):
                # Choose text color based on background
                text_color = 'white' if value > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', 
                       ha='center', va='center', 
                       color=text_color, fontsize=8, fontweight='bold')
    
    # Labels and title
    ax.set_xlabel('Age Group × Gender', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Race', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_title('Mean ASR Heatmap for All 126 Demographic Subgroups\nPGD Attack (ε=0.03)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(len(column_labels)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(RACES)) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  [OK] Saved heatmap: {os.path.basename(output_path)}")


def main():
    """Main execution"""
    print("\n" + "*" * 70)
    print("CREATING SUBGROUP ASR HEATMAP")
    print("*" * 70)
    
    # Get output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading aggregated ASR data...")
    df = load_aggregated_asr_data()
    print(f"  [OK] Loaded data for {len(df)} subgroups")
    
    # Create heatmap matrix
    print("\nCreating heatmap matrix...")
    heatmap_matrix, column_labels = create_heatmap_data(df)
    print(f"  [OK] Created matrix: {heatmap_matrix.shape[0]} races × {heatmap_matrix.shape[1]} age×gender combinations")
    
    # Count non-NaN values
    num_subgroups = np.sum(~np.isnan(heatmap_matrix))
    print(f"  [OK] Found data for {num_subgroups} subgroups")
    
    # Create and save heatmap
    print("\nGenerating heatmap...")
    output_path = os.path.join(output_dir, 'Subgroup_ASR_Heatmap_PGD_e0.03.png')
    create_heatmap(heatmap_matrix, column_labels, output_path)
    
    # Also save the matrix as CSV for reference
    matrix_df = pd.DataFrame(heatmap_matrix, index=RACES, columns=column_labels)
    csv_path = os.path.join(output_dir, 'Subgroup_ASR_Matrix_PGD_e0.03.csv')
    matrix_df.to_csv(csv_path)
    print(f"  [OK] Saved matrix CSV: {os.path.basename(csv_path)}")
    
    print("\n" + "*" * 70)
    print("HEATMAP GENERATION COMPLETE!")
    print("*" * 70)
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()

