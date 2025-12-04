"""
Script to create three perturbed versions of each demographic dataset
Each dataset will have versions for: C&W, PGD, and FGSM perturbations

This script applies perturbations to each face in the dataset without requiring FaceNet.
You can later test these perturbed images with FaceNet to see if they cause misidentification.
"""

import os
import pickle
import numpy as np
from PIL import Image
import random

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

# Perturbation types
PERTURBATION_TYPES = ['CW', 'PGD', 'FGSM']

# Script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)


def load_dataset(race_name, dataset_dir=OUTPUT_DIR):
    """
    Load a demographic dataset from pickle file
    
    Args:
        race_name: Name of the race (e.g., 'Black', 'White')
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
    return dataset


def apply_fgsm_perturbation(pil_image, epsilon=0.01):
    """
    Apply FGSM-style perturbation to an image
    For now, uses random noise with FGSM-like characteristics
    
    Args:
        pil_image: PIL Image
        epsilon: Perturbation magnitude (0-1)
    
    Returns:
        Perturbed PIL Image
    """
    # Convert to numpy array
    img_array = np.array(pil_image, dtype=np.float32)
    
    # Create perturbation (random noise with sign-based pattern)
    # FGSM uses sign of gradient, so we simulate with structured noise
    noise = np.random.randn(*img_array.shape).astype(np.float32)
    noise = np.sign(noise) * epsilon * 255  # Scale to pixel range
    
    # Apply perturbation
    perturbed = img_array + noise
    
    # Clamp to valid pixel range [0, 255]
    perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
    
    return Image.fromarray(perturbed)


def apply_pgd_perturbation(pil_image, epsilon=0.01, num_iter=10):
    """
    Apply PGD-style perturbation to an image
    Iterative perturbation with projection
    
    Args:
        pil_image: PIL Image
        epsilon: Maximum perturbation magnitude (0-1)
        num_iter: Number of iterations
    
    Returns:
        Perturbed PIL Image
    """
    # Convert to numpy array
    img_array = np.array(pil_image, dtype=np.float32)
    original = img_array.copy()
    
    # Iteratively build perturbation
    perturbation = np.zeros_like(img_array, dtype=np.float32)
    alpha = epsilon / num_iter  # Step size
    
    for _ in range(num_iter):
        # Add small random step
        step = np.random.randn(*img_array.shape).astype(np.float32)
        step = np.sign(step) * alpha * 255
        
        # Update perturbation
        perturbation += step
        
        # Project back to epsilon-ball
        perturbation_norm = np.linalg.norm(perturbation)
        if perturbation_norm > epsilon * 255:
            perturbation = (perturbation / perturbation_norm) * epsilon * 255
    
    # Apply perturbation
    perturbed = original + perturbation
    
    # Clamp to valid pixel range
    perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
    
    return Image.fromarray(perturbed)


def apply_cw_perturbation(pil_image, epsilon=0.01):
    """
    Apply C&W-style perturbation to an image
    Optimization-based perturbation (simplified version)
    
    Args:
        pil_image: PIL Image
        epsilon: Perturbation magnitude (0-1)
    
    Returns:
        Perturbed PIL Image
    """
    # Convert to numpy array
    img_array = np.array(pil_image, dtype=np.float32)
    
    # C&W uses L2 norm constraint, so create smooth perturbation
    # Create a smoother, more structured perturbation
    h, w, c = img_array.shape
    
    # Create low-frequency perturbation (smoother than random noise)
    # This simulates the optimization-based nature of C&W
    x = np.linspace(0, 2*np.pi, w)
    y = np.linspace(0, 2*np.pi, h)
    X, Y = np.meshgrid(x, y)
    
    # Combine multiple frequencies for smooth perturbation
    perturbation = np.zeros_like(img_array, dtype=np.float32)
    for freq in [1, 2, 3]:
        for channel in range(c):
            wave = np.sin(freq * X) * np.cos(freq * Y)
            perturbation[:, :, channel] = wave
    
    # Normalize and scale
    perturbation = perturbation / np.max(np.abs(perturbation))
    perturbation = perturbation * epsilon * 255
    
    # Apply perturbation
    perturbed = img_array + perturbation
    
    # Clamp to valid pixel range
    perturbed = np.clip(perturbed, 0, 255).astype(np.uint8)
    
    return Image.fromarray(perturbed)


def apply_perturbation(image_data, perturbation_type):
    """
    Apply a perturbation to a single image
    
    Args:
        image_data: Dictionary containing image and metadata
        perturbation_type: Type of perturbation ('CW', 'PGD', 'FGSM')
    
    Returns:
        Dictionary with perturbed image and same metadata
    """
    pil_image = image_data['image']
    
    # Apply appropriate perturbation
    if perturbation_type == 'FGSM':
        perturbed_image = apply_fgsm_perturbation(pil_image)
    elif perturbation_type == 'PGD':
        perturbed_image = apply_pgd_perturbation(pil_image)
    elif perturbation_type == 'CW':
        perturbed_image = apply_cw_perturbation(pil_image)
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")
    
    # Create new image data with perturbed image
    perturbed_data = image_data.copy()
    perturbed_data['image'] = perturbed_image
    perturbed_data['perturbation_type'] = perturbation_type
    
    return perturbed_data


def create_perturbed_dataset(race_name, perturbation_type, dataset_dir=OUTPUT_DIR):
    """
    Create a perturbed version of a demographic dataset
    Applies perturbation to each face in the dataset
    
    Args:
        race_name: Name of the race
        perturbation_type: Type of perturbation ('CW', 'PGD', 'FGSM')
        dataset_dir: Directory containing original datasets
    
    Returns:
        Dictionary with race name and list of perturbed image data
    """
    print(f"\n{'=' * 70}")
    print(f"Creating {perturbation_type} perturbed dataset for {race_name}")
    print(f"{'=' * 70}")
    
    # Load original dataset
    original_dataset = load_dataset(race_name, dataset_dir)
    race_data = original_dataset[race_name]
    
    # Create perturbed dataset
    perturbed_data = []
    
    for idx, image_data in enumerate(race_data):
        print(f"Processing image {idx + 1}/{len(race_data)}...", end='\r')
        
        # Apply perturbation to this image
        perturbed_image_data = apply_perturbation(image_data, perturbation_type)
        
        perturbed_data.append(perturbed_image_data)
    
    print(f"\n✓ Completed {perturbation_type} perturbation for {race_name}")
    
    # Return in same format as original
    return {race_name: perturbed_data}


def save_perturbed_dataset(perturbed_dataset, race_name, perturbation_type, output_dir=OUTPUT_DIR):
    """
    Save perturbed dataset to pickle file
    
    Args:
        perturbed_dataset: Dictionary with race name and perturbed image data
        race_name: Name of the race
        perturbation_type: Type of perturbation
        output_dir: Output directory
    """
    filename = f'{race_name}_{perturbation_type}.pkl'
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'wb') as f:
        pickle.dump(perturbed_dataset, f)
    
    print(f"✓ Saved perturbed dataset: {filepath}")


def create_all_perturbed_datasets(dataset_dir=OUTPUT_DIR, output_dir=OUTPUT_DIR):
    """
    Create all perturbed versions of all demographic datasets
    
    Args:
        dataset_dir: Directory containing original .pkl files
        output_dir: Directory to save perturbed .pkl files
    """
    print("\n" + "*" * 70)
    print("CREATING PERTURBED DATASETS")
    print("*" * 70)
    print(f"Races: {len(RACES)}")
    print(f"Perturbation types: {PERTURBATION_TYPES}")
    print(f"Total datasets to create: {len(RACES) * len(PERTURBATION_TYPES)}")
    print()
    
    total_created = 0
    
    for race in RACES:
        for perturbation_type in PERTURBATION_TYPES:
            try:
                # Create perturbed dataset
                perturbed_dataset = create_perturbed_dataset(
                    race, 
                    perturbation_type, 
                    dataset_dir
                )
                
                # Save perturbed dataset
                save_perturbed_dataset(
                    perturbed_dataset, 
                    race, 
                    perturbation_type, 
                    output_dir
                )
                
                total_created += 1
                
            except Exception as e:
                print(f"✗ Error processing {race} with {perturbation_type}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    print("\n" + "*" * 70)
    print("PERTURBED DATASET CREATION COMPLETE!")
    print("*" * 70)
    print(f"\nCreated {total_created} perturbed datasets")
    print(f"  - {len(PERTURBATION_TYPES)} versions for each of {len(RACES)} races")
    print(f"Output directory: {output_dir}")
    print("\nNext steps:")
    print("1. Create benchmark pairs (same race, gender, age range)")
    print("2. Test with FaceNet to verify baseline (faces should be identified as different)")
    print("3. Test again with perturbed versions to see if FaceNet misidentifies them")
    print()


def main():
    """
    Main execution function
    """
    print("\n" + "*" * 70)
    print("PERTURBED DATASET CREATION SCRIPT")
    print("*" * 70)
    print("\nThis script creates perturbed versions of each demographic dataset.")
    print("No FaceNet model is required - perturbations are applied directly to images.")
    print("You can test these perturbed datasets with FaceNet later.")
    print()
    
    # Create all perturbed datasets
    create_all_perturbed_datasets()


if __name__ == "__main__":
    main()
