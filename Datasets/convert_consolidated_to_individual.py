"""
Convert consolidated results_fairface_attacks.csv into individual CSV files
for each attack type and epsilon value.

This script reads the consolidated CSV file and creates 9 individual files:
- FGSM_e0.01_data.csv, FGSM_e0.03_data.csv, FGSM_e0.05_data.csv
- PGD_e0.01_data.csv, PGD_e0.03_data.csv, PGD_e0.05_data.csv
- CW_e0.01_data.csv, CW_e0.03_data.csv, CW_e0.05_data.csv
"""

import os
import pandas as pd
import numpy as np
from config import ATTACK_TYPES, EPSILONS
import sys
import os
# Add parent directory to path to import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_consolidated_file(consolidated_path='results_fairface_attacks.csv'):
    """
    Convert consolidated CSV to individual files for each attack/epsilon combination
    
    Args:
        consolidated_path: Path to the consolidated results_fairface_attacks.csv file
    """
    print("\n" + "*" * 70)
    print("CONVERTING CONSOLIDATED CSV TO INDIVIDUAL FILES")
    print("*" * 70)
    
    # Resolve absolute path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not os.path.isabs(consolidated_path):
        # Consolidated file is in the parent directory (project root)
        consolidated_path = os.path.join(os.path.dirname(script_dir), consolidated_path)
    
    if not os.path.exists(consolidated_path):
        raise FileNotFoundError(f"Consolidated file not found: {consolidated_path}")
    
    print(f"\nLoading consolidated file: {consolidated_path}")
    df = pd.read_csv(consolidated_path)
    print(f"  [OK] Loaded {len(df)} rows")
    
    # Create output directories for each attack/epsilon
    for attack_type in ATTACK_TYPES:
        for epsilon in EPSILONS:
            eps_dir = os.path.join('datasets', 'results', attack_type, f'e{epsilon:.2f}')
            os.makedirs(eps_dir, exist_ok=True)
    
    # Mapping from consolidated column names to expected format
    # Consolidated format: race, gender, age_range, img1, img2, d_clean, baseline_same, ...
    # Expected format: pair_id, race, gender, age, image_A_index, image_B_index, ...
    
    # Extract base columns
    base_cols = ['race', 'gender', 'age_range']
    
    # Mapping for attack types and their column prefixes
    attack_mappings = {
        'FGSM': {
            '001': ('d_fgsm_001', 'fgsm_same_001', 'fgsm_success_001'),
            '003': ('d_fgsm_003', 'fgsm_same_003', 'fgsm_success_003'),
            '005': ('d_fgsm_005', 'fgsm_same_005', 'fgsm_success_005'),
        },
        'PGD': {
            '001': ('d_pgd_001', 'pgd_same_001', 'pgd_success_001'),
            '003': ('d_pgd_003', 'pgd_same_003', 'pgd_success_003'),
            '005': ('d_pgd_005', 'pgd_same_005', 'pgd_success_005'),
        },
        'CW': {
            '001': ('d_cw', 'cw_same', 'cw_success'),  # Note: CW only has one set of columns
            '003': ('d_cw', 'cw_same', 'cw_success'),
            '005': ('d_cw', 'cw_same', 'cw_success'),
        }
    }
    
    epsilon_map = {
        0.01: '001',
        0.03: '003',
        0.05: '005'
    }
    
    files_created = []
    
    for attack_type in ATTACK_TYPES:
        for epsilon in EPSILONS:
            eps_key = epsilon_map[epsilon]
            
            # Get column names for this attack/epsilon
            if attack_type == 'CW':
                # CW uses the same columns for all epsilons
                dist_col, same_col, success_col = attack_mappings[attack_type]['001']
            else:
                dist_col, same_col, success_col = attack_mappings[attack_type][eps_key]
            
            # Check if columns exist
            required_cols = [dist_col, same_col, success_col]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                print(f"\n⚠ Skipping {attack_type}_e{epsilon:.2f}: Missing columns {missing_cols}")
                continue
            
            # Create output DataFrame
            output_df = pd.DataFrame()
            
            # Copy base columns
            output_df['race'] = df['race']
            output_df['gender'] = df['gender']
            output_df['age'] = df['age_range']  # Rename age_range to age
            
            # Extract image indices from paths (img1 and img2)
            # Format: "Race\Race_Gender_Age_XXX.jpg"
            def extract_index(img_path):
                if pd.isna(img_path):
                    return None
                # Get filename and extract number
                filename = os.path.basename(str(img_path))
                # Extract number from filename like "Black_Female_0-2_005.jpg" -> 5
                try:
                    # Find the number before .jpg
                    parts = filename.replace('.jpg', '').split('_')
                    if len(parts) >= 4:
                        num_str = parts[-1]
                        return int(num_str)
                except:
                    pass
                return None
            
            # For now, use a simple pair_id based on row index
            output_df['pair_id'] = df.apply(
                lambda row: f"{row['race']}_{row['gender']}_{row['age_range']}_{row.name:06d}",
                axis=1
            )
            
            # Extract indices from image paths
            output_df['image_A_index'] = df['img1'].apply(extract_index)
            output_df['image_B_index'] = df['img2'].apply(extract_index)
            
            # Copy gender and age for both images (they're the same in baseline pairs)
            output_df['image_A_gender'] = df['gender']
            output_df['image_A_age'] = df['age_range']
            output_df['image_B_gender'] = df['gender']
            output_df['image_B_age'] = df['age_range']
            
            # Add attack-specific columns
            output_df['epsilon'] = epsilon
            output_df['attack_type'] = attack_type
            output_df['baseline_distance'] = df['d_clean']
            output_df['adversarial_distance'] = df[dist_col]
            output_df['distance_change'] = df[dist_col] - df['d_clean']
            
            # Attack success: True if fgsm_success_001 is True
            output_df['Attack_Success_Status'] = df[success_col].astype(int)
            
            # Baseline false accept: True if baseline_same is True (distance < threshold)
            output_df['baseline_is_false_accept'] = df['baseline_same'].astype(bool)
            
            # Save to file
            filename = f'{attack_type}_e{epsilon:.2f}_data.csv'
            filepath = os.path.join('datasets', 'results', attack_type, f'e{epsilon:.2f}', filename)
            output_df.to_csv(filepath, index=False)
            
            files_created.append(filename)
            
            # Calculate ASR
            asr = output_df['Attack_Success_Status'].mean()
            print(f"  ✓ Created {filename}: {len(output_df)} pairs, ASR={asr:.2%}")
    
    print("\n" + "=" * 70)
    print(f"CONVERSION COMPLETE!")
    print("=" * 70)
    print(f"\nCreated {len(files_created)} files:")
    for f in files_created:
        print(f"  - {f}")
    print(f"\nFiles saved to: datasets/results/")
    
    return files_created


if __name__ == "__main__":
    convert_consolidated_file()

