"""
Adversarial Data Generation Script

This script generates adversarial examples by running Impersonation Attacks
on the baseline pairs. For each pair, it perturbs image A to minimize
distance to image B, then tests if the attack succeeded.

Goal: Create 9 unique perturbed datasets (3 attacks × 3 epsilons)
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from PIL import Image
import pickle
import sys

# Import configuration
from config import (
    ATTACK_TYPES, EPSILONS, DISTANCE_THRESHOLD, PGD_STEPS,
    BASELINE_DIR, ADVERSARIAL_DATA_DIR, RACES
)

# Import FaceNet model loader
# Ensure we can find FaceNet_Model.py in the parent directory
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Verify FaceNet_Model exists
facenet_path = os.path.join(parent_dir, 'FaceNet_Model.py')
if not os.path.exists(facenet_path):
    raise FileNotFoundError(
        f"FaceNet_Model.py not found at: {facenet_path}\n"
        f"Please ensure you're running this script from the Datasets/ directory\n"
        f"and that FaceNet_Model.py exists in the parent directory."
    )

from FaceNet_Model import get_facenet_model, get_face_embedding

# Import attack functions from CreatePerturbedDatasets in the same directory
# Use absolute import to avoid conflicts with files in parent directory
import importlib.util
perturbed_datasets_path = os.path.join(script_dir, 'CreatePerturbedDatasets.py')
if not os.path.exists(perturbed_datasets_path):
    raise FileNotFoundError(f"CreatePerturbedDatasets.py not found at: {perturbed_datasets_path}")

spec = importlib.util.spec_from_file_location("CreatePerturbedDatasets", perturbed_datasets_path)
CreatePerturbedDatasets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(CreatePerturbedDatasets)

# Import the functions we need
preprocess_image = CreatePerturbedDatasets.preprocess_image
fgsm_attack = CreatePerturbedDatasets.fgsm_attack
pgd_attack = CreatePerturbedDatasets.pgd_attack
cw_attack = CreatePerturbedDatasets.cw_attack
postprocess_image = CreatePerturbedDatasets.postprocess_image


def load_baseline_pairs():
    """Load all baseline pairs"""
    pairs_path = os.path.join(BASELINE_DIR, 'all_races_pairs.csv')
    
    if not os.path.exists(pairs_path):
        raise FileNotFoundError(f"Baseline pairs not found: {pairs_path}")
    
    pairs_df = pd.read_csv(pairs_path)
    print(f"[OK] Loaded {len(pairs_df)} baseline pairs")
    return pairs_df


def load_original_image(race_name, image_index):
    """Load original image from dataset"""
    dataset_path = os.path.join(os.path.dirname(__file__), f'{race_name}.pkl')
    
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    with open(dataset_path, 'rb') as f:
        dataset = pickle.load(f)
    
    if race_name not in dataset:
        raise KeyError(f"Race '{race_name}' not found in dataset")
    
    race_data = dataset[race_name]
    for img_data in race_data:
        if img_data['index'] == image_index:
            # Validate image data
            if 'image' not in img_data:
                raise ValueError(f"Image data missing 'image' key for index {image_index}")
            if img_data['image'] is None:
                raise ValueError(f"Image is None for index {image_index}")
            return img_data
    
    raise ValueError(f"Image {image_index} not found in {race_name} dataset")


def impersonation_attack(model, image_A_data, image_B_data, attack_type, epsilon):
    """
    Perform impersonation attack: perturb image A to minimize distance to image B
    
    Args:
        model: FaceNet model
        image_A_data: Dictionary with 'image' (PIL Image) - will be perturbed
        image_B_data: Dictionary with 'image' (PIL Image) - target to match
        attack_type: 'FGSM', 'PGD', or 'CW'
        epsilon: Perturbation budget
    
    Returns:
        perturbed_image_A: Perturbed PIL Image
        distance: Final distance between perturbed A and B
        attack_success: Boolean (True if distance < threshold)
    """
    # Validate images
    img_A = image_A_data['image']
    img_B = image_B_data['image']
    
    if img_A is None or img_B is None:
        raise ValueError("One or both images are None")
    
    if not hasattr(img_A, 'size') or not hasattr(img_B, 'size'):
        raise ValueError("Images are not valid PIL Images")
    
    if img_A.size[0] == 0 or img_A.size[1] == 0:
        raise ValueError(f"Image A has invalid size: {img_A.size}")
    if img_B.size[0] == 0 or img_B.size[1] == 0:
        raise ValueError(f"Image B has invalid size: {img_B.size}")
    
    # Ensure images are in RGB mode
    if img_A.mode != 'RGB':
        img_A = img_A.convert('RGB')
    if img_B.mode != 'RGB':
        img_B = img_B.convert('RGB')
    
    # Preprocess images to tensors
    image_A_tensor = preprocess_image(img_A)
    
    # Get target embedding (what we want image A to match)
    # Get embedding from image B (target) - use PIL Image, not tensor
    try:
        target_embedding = get_face_embedding(model, img_B)
    except Exception as e:
        raise ValueError(f"Failed to get embedding from image B: {e}")
    
    # Validate embedding
    if target_embedding is None:
        raise ValueError("Target embedding is None - face detection may have failed")
    
    # Handle list (keras-facenet sometimes returns list)
    if isinstance(target_embedding, list):
        if len(target_embedding) == 0 or target_embedding[0] is None:
            raise ValueError("Target embedding list is empty or contains None")
        target_embedding = target_embedding[0]
    
    # Convert to tensor (needed for gradient computation in attacks)
    # The embedding from PIL Image will be numpy array, convert to tensor
    if isinstance(target_embedding, tf.Tensor):
        # Already a tensor, but ensure it's float32
        target_embedding = tf.cast(target_embedding, dtype=tf.float32)
    elif isinstance(target_embedding, np.ndarray):
        # Convert numpy to tensor
        target_embedding = tf.convert_to_tensor(target_embedding, dtype=tf.float32)
    else:
        raise ValueError(f"Target embedding has unexpected type: {type(target_embedding)}")
    
    # Ensure proper shape: should be (1, 512) or (512,)
    # For loss calculation, we'll handle shape matching in the attack function
    # Keep as (1, 512) for now - will be squeezed in attack function if needed
    
    # Apply attack to minimize distance to target
    if attack_type == 'FGSM':
        perturbed_tensor = fgsm_attack(model, image_A_tensor, target_embedding, epsilon=epsilon)
    elif attack_type == 'PGD':
        perturbed_tensor = pgd_attack(model, image_A_tensor, target_embedding, 
                                     epsilon=epsilon, num_iter=PGD_STEPS)
    elif attack_type == 'CW':
        perturbed_tensor = cw_attack(model, image_A_tensor, target_embedding, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
    
    # Convert back to PIL Image
    perturbed_image = tensor_to_pil(perturbed_tensor)
    
    # Calculate final distance between perturbed A and original B
    try:
        perturbed_embedding = get_face_embedding(model, perturbed_image)
    except Exception as e:
        raise ValueError(f"Failed to get embedding from perturbed image: {e}")
    
    if perturbed_embedding is None:
        raise ValueError("Perturbed embedding is None - face detection may have failed")
    
    target_embedding_np = target_embedding.numpy() if isinstance(target_embedding, tf.Tensor) else target_embedding
    final_distance = calculate_distance(perturbed_embedding, target_embedding_np)
    
    # Check if attack succeeded (distance < threshold means model thinks they're the same)
    attack_success = final_distance < DISTANCE_THRESHOLD
    
    return perturbed_image, final_distance, attack_success


def tensor_to_pil(tensor):
    """Convert tensor to PIL Image"""
    # Use the postprocess_image function we imported
    return postprocess_image(tensor)


def calculate_distance(embedding_A, embedding_B):
    """Calculate squared Euclidean distance"""
    if isinstance(embedding_A, tf.Tensor):
        embedding_A = embedding_A.numpy()
    if isinstance(embedding_B, tf.Tensor):
        embedding_B = embedding_B.numpy()
    
    # Remove batch dimension if present
    if len(embedding_A.shape) > 1:
        embedding_A = embedding_A[0]
    if len(embedding_B.shape) > 1:
        embedding_B = embedding_B[0]
    
    return np.sum((embedding_A - embedding_B) ** 2)


def generate_adversarial_data(model, pairs_df, attack_type, epsilon):
    """
    Generate adversarial data for a specific attack type and epsilon
    
    Returns DataFrame with attack results
    """
    print(f"\n{'=' * 70}")
    print(f"Generating adversarial data: {attack_type} eps={epsilon}")
    print(f"{'=' * 70}")
    
    results = []
    total_pairs = len(pairs_df)
    
    for idx, pair in pairs_df.iterrows():
        if (idx + 1) % 100 == 0:
            print(f"  Processing pair {idx + 1}/{total_pairs}...", end='\r')
        
        try:
            # Load original images
            image_A = load_original_image(pair['race'], pair['image_A_index'])
            image_B = load_original_image(pair['race'], pair['image_B_index'])
            
            # Validate images before attack
            if 'image' not in image_A or image_A['image'] is None:
                raise ValueError(f"Image A is None for pair {pair['pair_id']}")
            if 'image' not in image_B or image_B['image'] is None:
                raise ValueError(f"Image B is None for pair {pair['pair_id']}")
            
            # Perform impersonation attack
            perturbed_A, final_distance, attack_success = impersonation_attack(
                model, image_A, image_B, attack_type, epsilon
            )
            
            # Store results
            results.append({
                'pair_id': pair['pair_id'],
                'race': pair['race'],
                'gender': pair['gender'],
                'age': pair['age'],
                'image_A_index': pair['image_A_index'],
                'image_B_index': pair['image_B_index'],
                'image_A_gender': pair['image_A_gender'],
                'image_A_age': pair['image_A_age'],
                'image_B_gender': pair['image_B_gender'],
                'image_B_age': pair['image_B_age'],
                'epsilon': epsilon,
                'attack_type': attack_type,
                'baseline_distance': pair['baseline_distance'],
                'adversarial_distance': final_distance,
                'distance_change': final_distance - pair['baseline_distance'],
                'Attack_Success_Status': 1 if attack_success else 0,
                'baseline_is_false_accept': pair['baseline_is_false_accept']
            })
            
        except Exception as e:
            print(f"\n  [ERROR] Error processing pair {pair['pair_id']}: {e}")
            continue
    
    print(f"\n  [OK] Processed {len(results)} pairs")
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate ASR
    if len(results_df) > 0:
        asr = results_df['Attack_Success_Status'].mean()
        print(f"  Attack Success Rate (ASR): {asr:.2%}")
    
    return results_df


def main():
    """Main execution"""
    print("\n" + "*" * 70)
    print("ADVERSARIAL DATA GENERATION")
    print("*" * 70)
    
    # Create output directory
    os.makedirs(ADVERSARIAL_DATA_DIR, exist_ok=True)
    
    # Load FaceNet model
    print("\nLoading FaceNet model...")
    model = get_facenet_model()
    
    if model is None:
        print("\n[ERROR] Cannot proceed without FaceNet model.")
        return
    
    # Load baseline pairs
    print("\nLoading baseline pairs...")
    pairs_df = load_baseline_pairs()
    
    # Generate adversarial data for each attack type and epsilon
    all_results = []
    
    for attack_type in ATTACK_TYPES:
        for epsilon in EPSILONS:
            try:
                results_df = generate_adversarial_data(model, pairs_df, attack_type, epsilon)
                
                # Save results
                filename = f'{attack_type}_e{epsilon:.2f}_data.csv'
                filepath = os.path.join(ADVERSARIAL_DATA_DIR, filename)
                results_df.to_csv(filepath, index=False)
                print(f"  [OK] Saved: {filename}")
                
                all_results.append(results_df)
                
            except Exception as e:
                print(f"[ERROR] Error generating {attack_type} eps={epsilon}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save combined results
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(ADVERSARIAL_DATA_DIR, 'all_adversarial_data.csv')
        combined_df.to_csv(combined_path, index=False)
        print(f"\n[OK] Saved combined results: {combined_path}")
    
    print("\n" + "*" * 70)
    print("ADVERSARIAL DATA GENERATION COMPLETE!")
    print("*" * 70)
    print(f"\nGenerated {len(all_results)} adversarial datasets")
    print(f"Results saved to: {ADVERSARIAL_DATA_DIR}")


if __name__ == "__main__":
    main()

