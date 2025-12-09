"""
Baseline Statistics Script for Facial Recognition Fairness Study

This script calculates False Acceptance Rate (FAR/FSR) for each demographic group
by testing negative pairs (different people, same race/age/gender).

The script:
1. Loads each race dataset
2. Groups images by race, age, and gender
3. Generates all unique negative pairs within each group
4. Calculates FaceNet embeddings and distances
5. Determines false accepts (distance < threshold)
6. Calculates FSR per image and per demographic group
7. Saves CSV files with results
8. Creates comprehensive visualizations
"""

import os
import pickle
import numpy as np
import pandas as pd
import itertools
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
# Note: Plotly imports available for future interactive visualizations
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots

# Import FaceNet model loader
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from FaceNet_Model import get_facenet_model, get_face_embedding

# Define the races from the original dataset
RACES = [
    'East_Asian',
    'Indian',
    'Black',
    'White',
    'Middle_Eastern',
    'Latino_Hispanic',
    'Southeast_Asian'
]

# Age and Gender mappings
AGE_MAPPING = {
    0: '0-2',
    1: '3-9',
    2: '10-19',
    3: '20-29',
    4: '30-39',
    5: '40-49',
    6: '50-59',
    7: '60-69',
    8: '70+'
}

GENDER_MAPPING = {
    0: 'Male',
    1: 'Female'
}

# FaceNet threshold (typical value: 1.0 to 1.2)
THRESHOLD = 1.0

# Color palette for races (consistent across all visualizations)
RACE_COLORS = {
    'East_Asian': '#FF6B6B',      # Red
    'Indian': '#4ECDC4',          # Teal
    'Black': '#45B7D1',           # Blue
    'White': '#FFA07A',           # Light Salmon
    'Middle_Eastern': '#98D8C8',  # Mint
    'Latino_Hispanic': '#F7DC6F', # Yellow
    'Southeast_Asian': '#BB8FCE'  # Purple
}

# Script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR
BASELINE_DIR = os.path.join(OUTPUT_DIR, 'baseline')


def preprocess_image(pil_image, target_size=(160, 160)):
    """
    Preprocess PIL image for FaceNet input.
    
    Args:
        pil_image: PIL Image
        target_size: Target size (width, height)
    
    Returns:
        Preprocessed image tensor ready for FaceNet
    """
    # Convert to RGB if needed
    if pil_image.mode != 'RGB':
        pil_image = pil_image.convert('RGB')
    
    # Resize to FaceNet input size (160x160)
    pil_image = pil_image.resize(target_size, Image.LANCZOS)
    
    # Convert to numpy array and normalize to [0, 1]
    img_array = np.array(pil_image, dtype=np.float32) / 255.0
    
    # Add batch dimension: (1, 160, 160, 3)
    img_tensor = tf.expand_dims(img_array, axis=0)
    
    return img_tensor


def load_dataset(race_name, dataset_dir=OUTPUT_DIR):
    """
    Load a demographic dataset from pickle file
    
    Args:
        race_name: Name of the race
        dataset_dir: Directory containing the .pkl files
    
    Returns:
        Dictionary with race name as key and list of image data as value
    """
    pickle_path = os.path.join(dataset_dir, f'{race_name}.pkl')
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Dataset file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    
    print(f"✓ Loaded {race_name} dataset: {len(dataset[race_name])} images")
    return dataset[race_name]


def group_by_demographics(race_data):
    """
    Group images by age and gender for controlled negative pairing.
    
    Args:
        race_data: List of image data dictionaries
    
    Returns:
        Dictionary: {(age, gender): [image_data, ...]}
    """
    groups = {}
    for img_data in race_data:
        age = img_data['age']
        gender = img_data['gender']
        key = (age, gender)
        
        if key not in groups:
            groups[key] = []
        groups[key].append(img_data)
    
    return groups


def generate_negative_pairs(group_images):
    """
    Generate all unique negative pairs from a group of images.
    
    For N images, generates N(N-1)/2 unique pairs.
    
    Args:
        group_images: List of image data dictionaries
    
    Returns:
        List of tuples: [(img_data_A, img_data_B), ...]
    """
    # Use itertools.combinations to generate all unique pairs
    pairs = list(itertools.combinations(group_images, 2))
    return pairs


def calculate_embedding(model, image_data):
    """
    Calculate FaceNet embedding for an image.
    
    Args:
        model: FaceNet model (keras-facenet FaceNet object)
        image_data: Dictionary containing 'image' (PIL Image)
    
    Returns:
        Embedding vector (numpy array)
    """
    pil_image = image_data['image']
    
    # keras-facenet expects PIL Images or numpy arrays, not tensors
    # It handles its own preprocessing (face detection, alignment, resizing)
    # So we pass the PIL image directly
    embedding = get_face_embedding(model, pil_image)
    
    # Convert tensor to numpy if needed
    if isinstance(embedding, tf.Tensor):
        embedding = embedding.numpy()
    
    # Remove batch dimension if present
    if len(embedding.shape) > 1:
        embedding = embedding[0]
    
    return embedding


def calculate_distance(embedding_A, embedding_B):
    """
    Calculate squared Euclidean distance between two embeddings.
    
    Args:
        embedding_A: First embedding vector
        embedding_B: Second embedding vector
    
    Returns:
        Squared Euclidean distance
    """
    # Squared Euclidean distance
    distance = np.sum((embedding_A - embedding_B) ** 2)
    return distance


def is_false_accept(distance, threshold=THRESHOLD):
    """
    Determine if a negative pair is a false accept.
    
    False Accept: Model predicts "same person" (distance < threshold)
    for two different people.
    
    Args:
        distance: Calculated distance between embeddings
        threshold: Distance threshold (default: 1.0)
    
    Returns:
        Boolean: True if false accept, False otherwise
    """
    return distance < threshold


def calculate_baseline_stats(race_name, model, dataset_dir=OUTPUT_DIR):
    """
    Calculate baseline statistics (FSR) for a race.
    
    Args:
        race_name: Name of the race
        model: FaceNet model
        dataset_dir: Directory containing datasets
    
    Returns:
        tuple: (image_results_df, pair_results_df)
            - image_results_df: DataFrame with results for each image
            - pair_results_df: DataFrame with results for each pair (for reuse with perturbed images)
    """
    print(f"\n{'=' * 70}")
    print(f"Processing {race_name}")
    print(f"{'=' * 70}")
    
    # Load dataset
    race_data = load_dataset(race_name, dataset_dir)
    
    # Group by demographics
    groups = group_by_demographics(race_data)
    print(f"  Found {len(groups)} demographic groups")
    
    # Store results for each image
    image_results = []
    # Store results for each pair (with unique IDs for reuse)
    pair_results = []
    pair_id_counter = 0
    
    # Process each demographic group
    for (age, gender), group_images in groups.items():
        print(f"  Processing {age} {gender}: {len(group_images)} images")
        
        # Generate all unique negative pairs
        pairs = generate_negative_pairs(group_images)
        print(f"    Generated {len(pairs)} negative pairs")
        
        # Calculate embeddings for all images in this group (cache to avoid recomputation)
        embeddings_cache = {}
        for img_data in group_images:
            img_idx = img_data['index']
            if img_idx not in embeddings_cache:
                embeddings_cache[img_idx] = calculate_embedding(model, img_data)
        
        # Track false accepts per image
        image_false_accepts = {img_data['index']: 0 for img_data in group_images}
        image_total_pairs = {img_data['index']: 0 for img_data in group_images}
        
        # Process each pair
        for img_A, img_B in pairs:
            idx_A = img_A['index']
            idx_B = img_B['index']
            
            # Generate unique pair ID
            pair_id = f"{race_name}_{pair_id_counter:06d}"
            pair_id_counter += 1
            
            # Get embeddings
            emb_A = embeddings_cache[idx_A]
            emb_B = embeddings_cache[idx_B]
            
            # Calculate distance
            distance = calculate_distance(emb_A, emb_B)
            
            # Check if false accept
            is_fa = is_false_accept(distance, THRESHOLD)
            
            # Store pair information for reuse with perturbed images
            pair_results.append({
                'pair_id': pair_id,
                'race': race_name,
                'age': age,
                'gender': gender,
                'image_A_index': idx_A,
                'image_B_index': idx_B,
                'image_A_age': img_A['age'],
                'image_A_gender': img_A['gender'],
                'image_B_age': img_B['age'],
                'image_B_gender': img_B['gender'],
                'baseline_distance': distance,
                'baseline_is_false_accept': is_fa,
                'distance_threshold': THRESHOLD
            })
            
            # Update counts for both images
            image_false_accepts[idx_A] += int(is_fa)
            image_false_accepts[idx_B] += int(is_fa)
            image_total_pairs[idx_A] += 1
            image_total_pairs[idx_B] += 1
        
        # Calculate FSR for each image in this group
        for img_data in group_images:
            idx = img_data['index']
            false_accepts = image_false_accepts[idx]
            total_pairs = image_total_pairs[idx]
            
            # FSR = False Accepts / Total Pairs
            fsr = false_accepts / total_pairs if total_pairs > 0 else 0.0
            
            image_results.append({
                'race': race_name,
                'index': idx,
                'age': age,
                'gender': gender,
                'false_accepts': false_accepts,
                'total_pairs': total_pairs,
                'fsr': fsr,
                'distance_threshold': THRESHOLD
            })
    
    # Create DataFrames
    image_df = pd.DataFrame(image_results)
    pair_df = pd.DataFrame(pair_results)
    
    print(f"  ✓ Processed {len(image_df)} images")
    print(f"  ✓ Generated {len(pair_df)} unique pairs")
    print(f"  Average FSR: {image_df['fsr'].mean():.4f}")
    
    return image_df, pair_df


def save_baseline_csv(image_df, pair_df, race_name, output_dir=BASELINE_DIR):
    """
    Save baseline statistics to CSV files.
    
    Args:
        image_df: DataFrame with results for each image
        pair_df: DataFrame with results for each pair
        race_name: Name of the race
        output_dir: Output directory
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save image-level CSV
    image_csv_path = os.path.join(output_dir, f'{race_name}_baseline.csv')
    image_df.to_csv(image_csv_path, index=False)
    print(f"  ✓ Saved image CSV: {image_csv_path}")
    
    # Save pair-level CSV (for reuse with perturbed images)
    pair_csv_path = os.path.join(output_dir, f'{race_name}_pairs.csv')
    pair_df.to_csv(pair_csv_path, index=False)
    print(f"  ✓ Saved pair CSV: {pair_csv_path}")
    
    # Also save as pickle for easier loading later
    pair_pkl_path = os.path.join(output_dir, f'{race_name}_pairs.pkl')
    pair_df.to_pickle(pair_pkl_path)
    print(f"  ✓ Saved pair pickle: {pair_pkl_path}")


def create_visualizations(all_results_df, output_dir=BASELINE_DIR):
    """
    Create comprehensive visualizations of baseline statistics.
    
    Args:
        all_results_df: DataFrame with results from all races
        output_dir: Output directory for saving plots
    """
    print(f"\n{'=' * 70}")
    print("Creating Visualizations")
    print(f"{'=' * 70}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. FSR by Gender for Each Race (Bar Plot)
    print("  1. Creating FSR by Gender bar plot...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    gender_race_fsr = all_results_df.groupby(['race', 'gender'])['fsr'].mean().reset_index()
    
    x = np.arange(len(RACES))
    width = 0.35
    
    # Get FSR values for each race and gender
    male_fsr = []
    female_fsr = []
    for race in RACES:
        race_gender_data = gender_race_fsr[gender_race_fsr['race'] == race]
        male_data = race_gender_data[race_gender_data['gender'] == 'Male']
        female_data = race_gender_data[race_gender_data['gender'] == 'Female']
        
        male_fsr.append(male_data['fsr'].values[0] if len(male_data) > 0 else 0)
        female_fsr.append(female_data['fsr'].values[0] if len(female_data) > 0 else 0)
    
    bars1 = ax.bar(x - width/2, male_fsr, width, label='Male', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, female_fsr, width, label='Female', color='#e74c3c', alpha=0.8)
    
    ax.set_xlabel('Race', fontsize=12, fontweight='bold')
    ax.set_ylabel('False Acceptance Rate (FSR)', fontsize=12, fontweight='bold')
    ax.set_title('FSR by Gender for Each Race', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(RACES, rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fsr_by_gender_per_race.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     ✓ Saved: fsr_by_gender_per_race.png")
    
    # 2. Male vs Female Overall Comparison
    print("  2. Creating Male vs Female comparison...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    gender_fsr = all_results_df.groupby('gender')['fsr'].mean()
    
    bars = ax.bar(gender_fsr.index, gender_fsr.values, 
                  color=['#3498db', '#e74c3c'], alpha=0.8)
    
    ax.set_xlabel('Gender', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average False Acceptance Rate (FSR)', fontsize=12, fontweight='bold')
    ax.set_title('Overall FSR: Male vs Female', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'male_vs_female_fsr.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     ✓ Saved: male_vs_female_fsr.png")
    
    # 3. Total FSR by Race
    print("  3. Creating total FSR by race bar plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    race_fsr = all_results_df.groupby('race')['fsr'].mean().sort_values(ascending=False)
    colors = [RACE_COLORS[race] for race in race_fsr.index]
    
    bars = ax.bar(race_fsr.index, race_fsr.values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_xlabel('Race', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average False Acceptance Rate (FSR)', fontsize=12, fontweight='bold')
    ax.set_title('Average FSR by Race', fontsize=14, fontweight='bold')
    ax.set_xticklabels(race_fsr.index, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'total_fsr_by_race.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     ✓ Saved: total_fsr_by_race.png")
    
    # 4. Gender Accuracy by Race (Bar Chart)
    print("  4. Creating gender accuracy bar chart for each race...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Calculate accuracy (1 - FSR) for each race and gender
    gender_race_accuracy = all_results_df.groupby(['race', 'gender'])['fsr'].mean().reset_index()
    gender_race_accuracy['accuracy'] = 1 - gender_race_accuracy['fsr']
    
    # Prepare data for grouped bar chart
    x = np.arange(len(RACES))
    width = 0.35
    
    male_accuracy = []
    female_accuracy = []
    for race in RACES:
        race_gender_data = gender_race_accuracy[gender_race_accuracy['race'] == race]
        male_data = race_gender_data[race_gender_data['gender'] == 'Male']
        female_data = race_gender_data[race_gender_data['gender'] == 'Female']
        
        male_accuracy.append(male_data['accuracy'].values[0] if len(male_data) > 0 else 0)
        female_accuracy.append(female_data['accuracy'].values[0] if len(female_data) > 0 else 0)
    
    # Create grouped bars
    bars1 = ax.bar(x - width/2, male_accuracy, width, label='Male', color='#3498db', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, female_accuracy, width, label='Female', color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1%}',
                    ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    ax.set_xlabel('Race', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy Rate', fontsize=12, fontweight='bold')
    ax.set_title('FaceNet Accuracy by Gender for Each Race\n'
                 '(Percentage of Correctly Identified Unique Faces)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(RACES, rotation=45, ha='right')
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1.1])  # Set y-axis to show 0-100%
    
    # Add key/legend box
    key_text = (
        "KEY:\n"
        "• Accuracy = 1 - FSR (False Acceptance Rate)\n"
        "• Distance ≥ 1.0: Correctly identified as DIFFERENT\n"
        "• Distance < 1.0: Incorrectly identified as SAME\n"
        "• Shows how accurately FaceNet identifies unique faces\n"
        "  for males vs females in each race"
    )
    ax.text(0.02, 0.98, key_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gender_accuracy_by_race.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     ✓ Saved: gender_accuracy_by_race.png")
    
    # 5. Scatterplot: FSR for Each Image (circles for women, triangles for men)
    print("  5. Creating scatterplot of FSR for each image...")
    fig, ax = plt.subplots(figsize=(16, 10))
    
    for race in RACES:
        race_data = all_results_df[all_results_df['race'] == race]
        color = RACE_COLORS[race]
        
        # Plot males (triangles)
        male_data = race_data[race_data['gender'] == 'Male']
        if len(male_data) > 0:
            ax.scatter(male_data['index'], male_data['fsr'], 
                      marker='^', s=100, alpha=0.6, color=color, 
                      label=f'{race} (Male)', edgecolors='black', linewidths=0.5)
        
        # Plot females (circles)
        female_data = race_data[race_data['gender'] == 'Female']
        if len(female_data) > 0:
            ax.scatter(female_data['index'], female_data['fsr'], 
                      marker='o', s=100, alpha=0.6, color=color, 
                      label=f'{race} (Female)', edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel('Image Index', fontsize=12, fontweight='bold')
    ax.set_ylabel('False Acceptance Rate (FSR)', fontsize=12, fontweight='bold')
    ax.set_title('FSR for Each Image by Race and Gender\n(Circles = Female, Triangles = Male)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2, fontsize=8)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'fsr_scatterplot_by_image.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"     ✓ Saved: fsr_scatterplot_by_image.png")
    
    # 6. Overall Summary Statistics CSV
    print("  6. Creating summary statistics...")
    summary_stats = []
    
    # Overall stats
    summary_stats.append({
        'group': 'Overall',
        'fsr_mean': all_results_df['fsr'].mean(),
        'fsr_std': all_results_df['fsr'].std(),
        'fsr_min': all_results_df['fsr'].min(),
        'fsr_max': all_results_df['fsr'].max(),
        'total_images': len(all_results_df)
    })
    
    # By race
    for race in RACES:
        race_data = all_results_df[all_results_df['race'] == race]
        summary_stats.append({
            'group': race,
            'fsr_mean': race_data['fsr'].mean(),
            'fsr_std': race_data['fsr'].std(),
            'fsr_min': race_data['fsr'].min(),
            'fsr_max': race_data['fsr'].max(),
            'total_images': len(race_data)
        })
    
    # By gender
    for gender in ['Male', 'Female']:
        gender_data = all_results_df[all_results_df['gender'] == gender]
        summary_stats.append({
            'group': gender,
            'fsr_mean': gender_data['fsr'].mean(),
            'fsr_std': gender_data['fsr'].std(),
            'fsr_min': gender_data['fsr'].min(),
            'fsr_max': gender_data['fsr'].max(),
            'total_images': len(gender_data)
        })
    
    summary_df = pd.DataFrame(summary_stats)
    summary_path = os.path.join(output_dir, 'summary_statistics.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"     ✓ Saved: summary_statistics.csv")
    
    print(f"\n✓ All visualizations saved to: {output_dir}")


def main():
    """
    Main execution function
    """
    print("\n" + "*" * 70)
    print("BASELINE STATISTICS CALCULATION")
    print("*" * 70)
    print("\nThis script calculates False Acceptance Rate (FSR) for each")
    print("demographic group using controlled negative pairing.")
    print(f"\nThreshold: {THRESHOLD}")
    print(f"Races: {len(RACES)}")
    print()
    
    # Load FaceNet model
    print("Loading FaceNet model...")
    model = get_facenet_model()
    
    if model is None:
        print("\n✗ Cannot proceed without FaceNet model.")
        print("\nPlease:")
        print("1. Install keras-facenet: pip install keras-facenet")
        print("   OR")
        print("2. Download FaceNet weights and place in models/ directory")
        return
    
    # Create baseline directory
    os.makedirs(BASELINE_DIR, exist_ok=True)
    
    # Process each race
    all_image_results = []
    all_pair_results = []
    
    for race in RACES:
        try:
            image_df, pair_df = calculate_baseline_stats(race, model)
            save_baseline_csv(image_df, pair_df, race)
            all_image_results.append(image_df)
            all_pair_results.append(pair_df)
        except Exception as e:
            print(f"✗ Error processing {race}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Combine all results
    if all_image_results:
        all_results_df = pd.concat(all_image_results, ignore_index=True)
        all_pairs_df = pd.concat(all_pair_results, ignore_index=True)
        
        # Save combined results
        combined_image_path = os.path.join(BASELINE_DIR, 'all_races_baseline.csv')
        all_results_df.to_csv(combined_image_path, index=False)
        print(f"\n✓ Saved combined image results: {combined_image_path}")
        
        combined_pair_path = os.path.join(BASELINE_DIR, 'all_races_pairs.csv')
        all_pairs_df.to_csv(combined_pair_path, index=False)
        print(f"✓ Saved combined pair results: {combined_pair_path}")
        
        combined_pair_pkl_path = os.path.join(BASELINE_DIR, 'all_races_pairs.pkl')
        all_pairs_df.to_pickle(combined_pair_pkl_path)
        print(f"✓ Saved combined pair pickle: {combined_pair_pkl_path}")
        
        # Create visualizations
        create_visualizations(all_results_df)
        
        # Print summary
        print("\n" + "*" * 70)
        print("BASELINE STATISTICS SUMMARY")
        print("*" * 70)
        print(f"\nOverall Average FSR: {all_results_df['fsr'].mean():.4f}")
        print(f"Overall FSR Std Dev: {all_results_df['fsr'].std():.4f}")
        print("\nFSR by Race:")
        race_summary = all_results_df.groupby('race')['fsr'].agg(['mean', 'std'])
        for race, row in race_summary.iterrows():
            print(f"  {race:<20}: {row['mean']:.4f} ± {row['std']:.4f}")
        print("\nFSR by Gender:")
        gender_summary = all_results_df.groupby('gender')['fsr'].agg(['mean', 'std'])
        for gender, row in gender_summary.iterrows():
            print(f"  {gender:<20}: {row['mean']:.4f} ± {row['std']:.4f}")
        print()
    else:
        print("\n✗ No results to process. Please check for errors above.")
    
    print("*" * 70)
    print("BASELINE STATISTICS COMPLETE!")
    print("*" * 70)
    print(f"\nResults saved to: {BASELINE_DIR}")
    print()


if __name__ == "__main__":
    main()

