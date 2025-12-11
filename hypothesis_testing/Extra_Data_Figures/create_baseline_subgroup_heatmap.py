"""
Create Heatmap of Mean FAR for All 126 Baseline Subgroups
(Race × Age × Gender combinations)

This script generates a heatmap showing the mean False Acceptance Rate (FAR)
for each of the 126 demographic subgroups from baseline pairs.
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

def load_baseline_data():
    """Load the baseline FSR data"""
    # Path to baseline data - try both lowercase and uppercase
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Try Datasets (uppercase) first, then datasets (lowercase)
    for datasets_dir in ['Datasets', 'datasets']:
        baseline_path = os.path.join(project_root, datasets_dir, 'baseline_stats', 'all_races_baseline.csv')
        if os.path.exists(baseline_path):
            break
    
    if not os.path.exists(baseline_path):
        raise FileNotFoundError(f"Baseline data file not found: {baseline_path}")
    
    df = pd.read_csv(baseline_path)
    return df


def aggregate_baseline_by_subgroups(df):
    """
    Aggregate baseline FSR by Race × Gender × Age subgroups
    
    Returns DataFrame with mean FSR for each subgroup
    """
    # Group by demographics and calculate mean FSR
    aggregated = df.groupby(['race', 'gender', 'age'])['fsr'].agg([
        'mean',  # Mean FSR (FAR)
        'count',  # Number of images
        'std'     # Standard deviation
    ]).reset_index()
    
    aggregated.columns = ['race', 'gender', 'age', 'mean_fsr', 'n_images', 'std_fsr']
    
    return aggregated


def create_heatmap_data(df):
    """
    Create a matrix for the heatmap
    
    Rows: Races (7)
    Columns: Age × Gender combinations (9 ages × 2 genders = 18)
    Values: Mean FSR (FAR)
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
                    heatmap_matrix[i, col_idx] = matching_rows.iloc[0]['mean_fsr']
    
    return heatmap_matrix, column_labels


def create_heatmap(heatmap_matrix, column_labels, output_path):
    """
    Create and save the heatmap with red-to-green color coding
    
    Red = High FAR (more vulnerable)
    Green = Low FAR (less vulnerable)
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
    cbar.set_label('Mean FAR', rotation=270, labelpad=20, fontsize=12, fontweight='bold')
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
    ax.set_title('Mean FAR Heatmap for All 126 Demographic Subgroups\nBaseline Pairs', 
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
    print("CREATING BASELINE SUBGROUP FAR HEATMAP")
    print("*" * 70)
    
    # Get output directory
    output_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print("\nLoading baseline data...")
    df = load_baseline_data()
    print(f"  [OK] Loaded {len(df)} baseline records")
    
    # Aggregate by subgroups
    print("\nAggregating FAR by subgroups...")
    aggregated_df = aggregate_baseline_by_subgroups(df)
    print(f"  [OK] Created {len(aggregated_df)} demographic subgroups")
    
    # Save aggregated data
    agg_path = os.path.join(output_dir, 'Baseline_aggregated_FAR_by_subgroups.csv')
    aggregated_df.to_csv(agg_path, index=False)
    print(f"  [OK] Saved aggregated data: {os.path.basename(agg_path)}")
    
    # Create heatmap matrix
    print("\nCreating heatmap matrix...")
    heatmap_matrix, column_labels = create_heatmap_data(aggregated_df)
    print(f"  [OK] Created matrix: {heatmap_matrix.shape[0]} races × {heatmap_matrix.shape[1]} age×gender combinations")
    
    # Count non-NaN values
    num_subgroups = np.sum(~np.isnan(heatmap_matrix))
    print(f"  [OK] Found data for {num_subgroups} subgroups")
    
    # Create and save heatmap
    print("\nGenerating heatmap...")
    output_path = os.path.join(output_dir, 'Baseline_Subgroup_FAR_Heatmap.png')
    create_heatmap(heatmap_matrix, column_labels, output_path)
    
    # Also save the matrix as CSV for reference
    matrix_df = pd.DataFrame(heatmap_matrix, index=RACES, columns=column_labels)
    csv_path = os.path.join(output_dir, 'Baseline_Subgroup_FAR_Matrix.csv')
    matrix_df.to_csv(csv_path)
    print(f"  [OK] Saved matrix CSV: {os.path.basename(csv_path)}")
    
    print("\n" + "*" * 70)
    print("BASELINE HEATMAP GENERATION COMPLETE!")
    print("*" * 70)
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()

