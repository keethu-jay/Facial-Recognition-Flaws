"""
Compare Perturbed Datasets to Baseline Statistics

This script:
1. Loads baseline pairs (from CreateBaselineStats.py)
2. Loads perturbed datasets (from CreatePerturbedDatasets.py)
3. Tests the same pairs with perturbed images
4. Calculates FSR for perturbed images
5. Compares baseline FSR vs perturbed FSR
6. Creates comparison visualizations
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Import FaceNet model loader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FaceNet_Model import get_facenet_model, get_face_embedding

# Define constants
RACES = [
    'East_Asian',
    'Indian',
    'Black',
    'White',
    'Middle_Eastern',
    'Latino_Hispanic',
    'Southeast_Asian'
]

PERTURBATION_TYPES = ['FGSM', 'PGD', 'CW']
EPSILON_VALUES = [0.01, 0.03, 0.05]
THRESHOLD = 1.0

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_DIR = os.path.join(SCRIPT_DIR, 'baseline')
PERTURBED_BASE_DIR = SCRIPT_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'perturbed_comparison')

# Color palette
RACE_COLORS = {
    'East_Asian': '#FF6B6B',
    'Indian': '#4ECDC4',
    'Black': '#45B7D1',
    'White': '#FFA07A',
    'Middle_Eastern': '#98D8C8',
    'Latino_Hispanic': '#F7DC6F',
    'Southeast_Asian': '#BB8FCE'
}


def preprocess_image(pil_image, target_size=(160, 160)):
    """Preprocess PIL image for FaceNet"""
    from PIL import Image
    import numpy as np
    
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    pil_image = pil_image.resize(target_size, Image.LANCZOS)
    img_array = np.array(pil_image, dtype=np.float32) / 255.0
    
    return img_array


def calculate_distance(embedding_A, embedding_B):
    """Calculate squared Euclidean distance"""
    return np.sum((embedding_A - embedding_B) ** 2)


def load_baseline_pairs(race_name):
    """Load baseline pairs for a race"""
    pairs_path = os.path.join(BASELINE_DIR, f'{race_name}_pairs.csv')
    
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Baseline pairs not found: {pairs_path}")
    
    pairs_df = pd.read_csv(pairs_path)
    print(f"  ✓ Loaded {len(pairs_df)} baseline pairs for {race_name}")
    return pairs_df


def load_perturbed_dataset(race_name, perturbation_type, epsilon):
    """Load perturbed dataset"""
    perturbed_path = os.path.join(
        PERTURBED_BASE_DIR, 
        perturbation_type, 
        race_name, 
        f'{race_name}_{epsilon:.2f}.pkl'
    )
    
    if not os.path.exists(perturbed_path):
        raise FileNotFoundError(f"Perturbed dataset not found: {perturbed_path}")
    
    with open(perturbed_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Convert to dictionary for easy lookup by index
    race_data = dataset[race_name]
    image_dict = {img_data['index']: img_data for img_data in race_data}
    
    return image_dict


def load_original_dataset(race_name):
    """Load original dataset"""
    original_path = os.path.join(PERTURBED_BASE_DIR, f'{race_name}.pkl')
    
    with open(original_path, 'rb') as f:
        dataset = pickle.load(f)
    
    race_data = dataset[race_name]
    image_dict = {img_data['index']: img_data for img_data in race_data}
    
    return image_dict


def test_perturbed_pairs(race_name, perturbation_type, epsilon, model, pairs_df):
    """
    Test perturbed images using baseline pairs
    
    Returns DataFrame with comparison results
    """
    print(f"    Testing {perturbation_type} ε={epsilon}...")
    
    # Load datasets
    perturbed_images = load_perturbed_dataset(race_name, perturbation_type, epsilon)
    original_images = load_original_dataset(race_name)
    
    results = []
    
    for _, pair in pairs_df.iterrows():
        pair_id = pair['pair_id']
        idx_A = pair['image_A_index']
        idx_B = pair['image_B_index']
        
        # Get images
        perturbed_img_A = perturbed_images[idx_A]
        original_img_B = original_images[idx_B]
        
        # Calculate embeddings
        img_A_array = preprocess_image(perturbed_img_A['image'])
        img_B_array = preprocess_image(original_img_B['image'])
        
        emb_A = get_face_embedding(model, perturbed_img_A['image'])
        emb_B = get_face_embedding(model, original_img_B['image'])
        
        # Calculate distance
        distance = calculate_distance(emb_A, emb_B)
        
        # Check if false accept
        is_false_accept = distance < THRESHOLD
        
        results.append({
            'pair_id': pair_id,
            'race': race_name,
            'perturbation_type': perturbation_type,
            'epsilon': epsilon,
            'image_A_index': idx_A,
            'image_B_index': idx_B,
            'image_A_gender': perturbed_img_A['gender'],
            'image_A_age': perturbed_img_A['age'],
            'baseline_distance': pair['baseline_distance'],
            'baseline_is_false_accept': pair['baseline_is_false_accept'],
            'perturbed_distance': distance,
            'perturbed_is_false_accept': is_false_accept,
            'distance_change': distance - pair['baseline_distance'],
            'attack_succeeded': pair['baseline_is_false_accept'] == False and is_false_accept == True
        })
    
    return pd.DataFrame(results)


def calculate_perturbed_fsr(comparison_df, race_name, perturbation_type, epsilon):
    """Calculate FSR for perturbed images"""
    # Group by image to calculate FSR per image
    image_results = []
    
    for image_idx in comparison_df['image_A_index'].unique():
        image_pairs = comparison_df[comparison_df['image_A_index'] == image_idx]
        
        false_accepts = image_pairs['perturbed_is_false_accept'].sum()
        total_pairs = len(image_pairs)
        fsr = false_accepts / total_pairs if total_pairs > 0 else 0.0
        
        # Get image metadata
        first_pair = image_pairs.iloc[0]
        
        image_results.append({
            'race': race_name,
            'index': image_idx,
            'perturbation_type': perturbation_type,
            'epsilon': epsilon,
            'age': first_pair['image_A_age'],
            'gender': first_pair['image_A_gender'],
            'false_accepts': false_accepts,
            'total_pairs': total_pairs,
            'fsr': fsr
        })
    
    return pd.DataFrame(image_results)


def compare_all_perturbed_to_baseline(model):
    """
    Compare all perturbed datasets to baseline
    
    Returns combined results DataFrame
    """
    print("\n" + "*" * 70)
    print("COMPARING PERTURBED DATASETS TO BASELINE")
    print("*" * 70)
    
    all_comparison_results = []
    all_fsr_results = []
    
    for race in RACES:
        print(f"\n{'=' * 70}")
        print(f"Processing {race}")
        print(f"{'=' * 70}")
        
        try:
            # Load baseline pairs
            pairs_df = load_baseline_pairs(race)
            
            # Test each perturbation type and epsilon
            for perturbation_type in PERTURBATION_TYPES:
                for epsilon in EPSILON_VALUES:
                    try:
                        # Test pairs with perturbed images
                        comparison_df = test_perturbed_pairs(
                            race, perturbation_type, epsilon, model, pairs_df
                        )
                        all_comparison_results.append(comparison_df)
                        
                        # Calculate FSR
                        fsr_df = calculate_perturbed_fsr(
                            comparison_df, race, perturbation_type, epsilon
                        )
                        all_fsr_results.append(fsr_df)
                        
                        # Print summary
                        attack_success_rate = comparison_df['attack_succeeded'].mean()
                        avg_distance_change = comparison_df['distance_change'].mean()
                        print(f"      Attack success rate: {attack_success_rate:.2%}")
                        print(f"      Avg distance change: {avg_distance_change:.4f}")
                        
                    except FileNotFoundError as e:
                        print(f"      ✗ Skipping: {e}")
                        continue
                    except Exception as e:
                        print(f"      ✗ Error: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                        
        except FileNotFoundError as e:
            print(f"  ✗ Skipping {race}: {e}")
            continue
        except Exception as e:
            print(f"  ✗ Error processing {race}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all results
    if all_comparison_results:
        combined_comparison = pd.concat(all_comparison_results, ignore_index=True)
        combined_fsr = pd.concat(all_fsr_results, ignore_index=True)
        
        return combined_comparison, combined_fsr
    else:
        return None, None


def create_comparison_visualizations(comparison_df, fsr_df, baseline_df):
    """Create visualizations comparing baseline to perturbed"""
    print("\n" + "*" * 70)
    print("Creating Comparison Visualizations")
    print("*" * 70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load baseline FSR for comparison
    baseline_fsr = baseline_df.groupby(['race', 'gender'])['fsr'].mean().reset_index()
    baseline_fsr['source'] = 'Baseline'
    
    # Calculate perturbed FSR by race, gender, perturbation, epsilon
    perturbed_fsr_summary = fsr_df.groupby([
        'race', 'gender', 'perturbation_type', 'epsilon'
    ])['fsr'].mean().reset_index()
    perturbed_fsr_summary['source'] = 'Perturbed'
    
    # 1. FSR Comparison: Baseline vs Perturbed by Race and Gender
    print("  1. Creating FSR comparison chart...")
    fig, axes = plt.subplots(len(PERTURBATION_TYPES), len(EPSILON_VALUES), 
                            figsize=(18, 12))
    
    for p_idx, pert_type in enumerate(PERTURBATION_TYPES):
        for e_idx, eps in enumerate(EPSILON_VALUES):
            ax = axes[p_idx, e_idx]
            
            # Get data for this perturbation and epsilon
            pert_data = perturbed_fsr_summary[
                (perturbed_fsr_summary['perturbation_type'] == pert_type) &
                (perturbed_fsr_summary['epsilon'] == eps)
            ]
            
            # Merge with baseline
            comparison_data = []
            for race in RACES:
                for gender in ['Male', 'Female']:
                    baseline_val = baseline_fsr[
                        (baseline_fsr['race'] == race) &
                        (baseline_fsr['gender'] == gender)
                    ]['fsr'].values
                    
                    perturbed_val = pert_data[
                        (pert_data['race'] == race) &
                        (pert_data['gender'] == gender)
                    ]['fsr'].values
                    
                    if len(baseline_val) > 0 and len(perturbed_val) > 0:
                        comparison_data.append({
                            'race': race,
                            'gender': gender,
                            'baseline_fsr': baseline_val[0],
                            'perturbed_fsr': perturbed_val[0],
                            'change': perturbed_val[0] - baseline_val[0]
                        })
            
            comp_df = pd.DataFrame(comparison_data)
            
            if len(comp_df) > 0:
                x = np.arange(len(RACES))
                width = 0.35
                
                male_baseline = [comp_df[(comp_df['race'] == r) & 
                                        (comp_df['gender'] == 'Male')]['baseline_fsr'].values[0] 
                                if len(comp_df[(comp_df['race'] == r) & 
                                              (comp_df['gender'] == 'Male')]) > 0 else 0 
                                for r in RACES]
                male_perturbed = [comp_df[(comp_df['race'] == r) & 
                                         (comp_df['gender'] == 'Male')]['perturbed_fsr'].values[0] 
                                 if len(comp_df[(comp_df['race'] == r) & 
                                               (comp_df['gender'] == 'Male')]) > 0 else 0 
                                 for r in RACES]
                
                ax.bar(x - width/2, male_baseline, width, label='Baseline (Male)', 
                      color='#3498db', alpha=0.6)
                ax.bar(x - width/2, male_perturbed, width, label='Perturbed (Male)', 
                      color='#3498db', alpha=0.9, bottom=male_baseline)
                
                ax.set_title(f'{pert_type} ε={eps}', fontweight='bold')
                ax.set_xticks(x)
                ax.set_xticklabels(RACES, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('FSR', fontsize=9)
                ax.legend(fontsize=7)
                ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('FSR Comparison: Baseline vs Perturbed by Attack Type and Epsilon', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fsr_comparison_all.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     ✓ Saved: fsr_comparison_all.png")
    
    # 2. Attack Success Rate by Race
    print("  2. Creating attack success rate chart...")
    attack_success = comparison_df.groupby([
        'race', 'perturbation_type', 'epsilon'
    ])['attack_succeeded'].mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for pert_type in PERTURBATION_TYPES:
        for eps in EPSILON_VALUES:
            data = attack_success[
                (attack_success['perturbation_type'] == pert_type) &
                (attack_success['epsilon'] == eps)
            ]
            
            label = f'{pert_type} ε={eps}'
            ax.plot(data['race'], data['attack_succeeded'], 
                   marker='o', label=label, linewidth=2, markersize=8)
    
    ax.set_xlabel('Race', fontsize=12, fontweight='bold')
    ax.set_ylabel('Attack Success Rate', fontsize=12, fontweight='bold')
    ax.set_title('Attack Success Rate by Race\n'
                '(Percentage of pairs where attack succeeded)', 
                fontsize=14, fontweight='bold')
    ax.set_xticklabels(attack_success['race'].unique(), rotation=45, ha='right')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'attack_success_rate.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     ✓ Saved: attack_success_rate.png")
    
    print(f"\n✓ All visualizations saved to: {OUTPUT_DIR}")


def main():
    """Main execution"""
    print("\n" + "*" * 70)
    print("PERTURBED DATASET COMPARISON TO BASELINE")
    print("*" * 70)
    
    # Load FaceNet model
    print("\nLoading FaceNet model...")
    model = get_facenet_model()
    
    if model is None:
        print("\n✗ Cannot proceed without FaceNet model.")
        return
    
    # Load baseline FSR data
    print("\nLoading baseline FSR data...")
    baseline_path = os.path.join(BASELINE_DIR, 'all_races_baseline.csv')
    baseline_df = pd.read_csv(baseline_path)
    print(f"✓ Loaded baseline data: {len(baseline_df)} images")
    
    # Compare perturbed to baseline
    comparison_df, fsr_df = compare_all_perturbed_to_baseline(model)
    
    if comparison_df is None or len(comparison_df) == 0:
        print("\n✗ No comparison results generated. Check for errors above.")
        return
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    comparison_path = os.path.join(OUTPUT_DIR, 'comparison_results.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"\n✓ Saved comparison results: {comparison_path}")
    
    fsr_path = os.path.join(OUTPUT_DIR, 'perturbed_fsr_results.csv')
    fsr_df.to_csv(fsr_path, index=False)
    print(f"✓ Saved perturbed FSR results: {fsr_path}")
    
    # Create visualizations
    create_comparison_visualizations(comparison_df, fsr_df, baseline_df)
    
    # Print summary
    print("\n" + "*" * 70)
    print("SUMMARY")
    print("*" * 70)
    
    overall_attack_success = comparison_df['attack_succeeded'].mean()
    print(f"\nOverall Attack Success Rate: {overall_attack_success:.2%}")
    
    print("\nAttack Success by Type:")
    for pert_type in PERTURBATION_TYPES:
        pert_data = comparison_df[comparison_df['perturbation_type'] == pert_type]
        success_rate = pert_data['attack_succeeded'].mean()
        print(f"  {pert_type}: {success_rate:.2%}")
    
    print("\n" + "*" * 70)
    print("COMPARISON COMPLETE!")
    print("*" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

