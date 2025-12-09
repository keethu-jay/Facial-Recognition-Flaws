"""
Baseline Evaluation Script for FaceNet Verification

This script evaluates FaceNet's verification accuracy on clean (unperturbed) images.
For each face in each dataset, it compares it with another face of the same gender
and age range. If the distance calculated is greater than 1.0, FaceNet sees them as
different people (correct for different people).

Metrics Calculated:
- Accuracy: Percentage of correct identifications (distance > 1.0 for different people)
- False Acceptance Rate (FAR): Percentage of different people incorrectly identified as same (distance <= 1.0)
- False Rejection Rate (FRR): Not applicable for different-person comparisons, but tracked for completeness

Per-group metrics are computed for each race, and overall metrics across all races.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from collections import defaultdict
import itertools
from datetime import datetime

# Import FaceNet model loader
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
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

# Script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASETS_DIR = os.path.join(SCRIPT_DIR, 'Datasets')
OUTPUT_DIR = SCRIPT_DIR

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Distance threshold for FaceNet verification
# If distance > THRESHOLD, faces are considered different (correct for different people)
# If distance <= THRESHOLD, faces are considered same (incorrect for different people)
THRESHOLD = 1.0


def load_dataset(race_name, dataset_dir=DATASETS_DIR):
    """
    Load a demographic dataset from pickle file
    
    Args:
        race_name: Name of the race (e.g., 'Black', 'White')
        dataset_dir: Directory containing the .pkl files
    
    Returns:
        List of image data dictionaries
    """
    pickle_path = os.path.join(dataset_dir, f'{race_name}.pkl')
    
    if not os.path.exists(pickle_path):
        raise FileNotFoundError(f"Dataset file not found: {pickle_path}")
    
    with open(pickle_path, 'rb') as f:
        dataset = pickle.load(f)
    
    # Extract the list of images from the dictionary
    images = dataset[race_name]
    print(f"✓ Loaded {race_name} dataset: {len(images)} images")
    return images


def preprocess_image(pil_image, target_size=(160, 160)):
    """
    Preprocess PIL image for FaceNet input.
    FaceNet expects images to be 160x160, normalized to [0, 1].
    
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


def calculate_distance(embedding1, embedding2):
    """
    Calculate Euclidean distance between two face embeddings.
    
    Args:
        embedding1: First face embedding vector
        embedding2: Second face embedding vector
    
    Returns:
        Euclidean distance (float)
    """
    # Convert to numpy if they're tensors
    if isinstance(embedding1, tf.Tensor):
        embedding1 = embedding1.numpy()
    if isinstance(embedding2, tf.Tensor):
        embedding2 = embedding2.numpy()
    
    # Flatten if needed
    if len(embedding1.shape) > 1:
        embedding1 = embedding1.flatten()
    if len(embedding2.shape) > 1:
        embedding2 = embedding2.flatten()
    
    # Calculate Euclidean distance
    distance = np.linalg.norm(embedding1 - embedding2)
    return float(distance)


def create_comparison_pairs(images):
    """
    Create pairs of images for comparison.
    Each image is paired with another image of the same gender and age range.
    
    Args:
        images: List of image data dictionaries
    
    Returns:
        List of tuples (image1, image2) where both have same gender and age
    """
    # Organize images by gender and age
    organized = defaultdict(list)
    for img in images:
        key = (img['gender'], img['age'])
        organized[key].append(img)
    
    pairs = []
    
    # For each (gender, age) group, create pairs
    for (gender, age), group_images in organized.items():
        # Create all possible pairs within the group
        for img1, img2 in itertools.combinations(group_images, 2):
            # Ensure we're comparing different people (different indices)
            if img1['index'] != img2['index']:
                pairs.append((img1, img2))
    
    return pairs


def evaluate_race_dataset(race_name, model, dataset_dir=DATASETS_DIR):
    """
    Evaluate FaceNet verification accuracy for a single race dataset.
    
    Args:
        race_name: Name of the race to evaluate
        model: FaceNet model
        dataset_dir: Directory containing dataset files
    
    Returns:
        Dictionary containing evaluation metrics and detailed results
    """
    print(f"\n{'='*70}")
    print(f"Evaluating {race_name} dataset")
    print(f"{'='*70}")
    
    # Load dataset
    images = load_dataset(race_name, dataset_dir)
    
    # Create comparison pairs (same gender, same age range)
    pairs = create_comparison_pairs(images)
    print(f"Created {len(pairs)} comparison pairs (same gender, same age)")
    
    if len(pairs) == 0:
        print(f"⚠ Warning: No valid pairs found for {race_name}")
        return None
    
    # Evaluate each pair
    distances = []
    correct_predictions = 0  # Distance > threshold (correctly identified as different)
    false_acceptances = 0     # Distance <= threshold (incorrectly identified as same)
    
    print("Computing embeddings and distances...")
    for i, (img1, img2) in enumerate(pairs):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(pairs)} pairs...")
        
        # Preprocess images
        img1_tensor = preprocess_image(img1['image'])
        img2_tensor = preprocess_image(img2['image'])
        
        # Get embeddings
        embedding1 = get_face_embedding(model, img1_tensor)
        embedding2 = get_face_embedding(model, img2_tensor)
        
        # Calculate distance
        distance = calculate_distance(embedding1, embedding2)
        distances.append(distance)
        
        # Evaluate: distance > threshold means correctly identified as different
        if distance > THRESHOLD:
            correct_predictions += 1
        else:
            false_acceptances += 1
    
    # Calculate metrics
    total_pairs = len(pairs)
    accuracy = correct_predictions / total_pairs if total_pairs > 0 else 0.0
    far = false_acceptances / total_pairs if total_pairs > 0 else 0.0
    
    # FRR is not applicable here since we're comparing different people
    # FRR would be relevant for same-person comparisons (false rejections)
    frr = 0.0  # Not applicable for different-person comparisons
    
    results = {
        'race': race_name,
        'total_images': len(images),
        'total_pairs': total_pairs,
        'distances': distances,
        'mean_distance': np.mean(distances),
        'std_distance': np.std(distances),
        'min_distance': np.min(distances),
        'max_distance': np.max(distances),
        'correct_predictions': correct_predictions,
        'false_acceptances': false_acceptances,
        'accuracy': accuracy,
        'far': far,
        'frr': frr,
        'threshold': THRESHOLD
    }
    
    print(f"\nResults for {race_name}:")
    print(f"  Total pairs: {total_pairs}")
    print(f"  Mean distance: {results['mean_distance']:.4f}")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  FAR: {far*100:.2f}%")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  False acceptances: {false_acceptances}")
    
    return results


def calculate_overall_metrics(all_results):
    """
    Calculate overall metrics across all races.
    
    Args:
        all_results: List of result dictionaries from each race
    
    Returns:
        Dictionary with overall metrics
    """
    total_pairs = sum(r['total_pairs'] for r in all_results)
    total_correct = sum(r['correct_predictions'] for r in all_results)
    total_false_acceptances = sum(r['false_acceptances'] for r in all_results)
    
    overall_accuracy = total_correct / total_pairs if total_pairs > 0 else 0.0
    overall_far = total_false_acceptances / total_pairs if total_pairs > 0 else 0.0
    
    # Calculate disparity (difference between best and worst performing groups)
    accuracies = [r['accuracy'] for r in all_results]
    max_accuracy = max(accuracies)
    min_accuracy = min(accuracies)
    accuracy_disparity = max_accuracy - min_accuracy
    
    far_values = [r['far'] for r in all_results]
    max_far = max(far_values)
    min_far = min(far_values)
    far_disparity = max_far - min_far
    
    return {
        'total_pairs': total_pairs,
        'total_correct': total_correct,
        'total_false_acceptances': total_false_acceptances,
        'overall_accuracy': overall_accuracy,
        'overall_far': overall_far,
        'overall_frr': 0.0,  # Not applicable
        'accuracy_disparity': accuracy_disparity,
        'far_disparity': far_disparity,
        'max_accuracy': max_accuracy,
        'min_accuracy': min_accuracy,
        'max_far': max_far,
        'min_far': min_far
    }


def generate_report(all_results, overall_metrics, output_path):
    """
    Generate a comprehensive evaluation report.
    
    Args:
        all_results: List of result dictionaries from each race
        overall_metrics: Dictionary with overall metrics
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("FaceNet Baseline Evaluation Report\n")
        f.write("="*70 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Threshold: {THRESHOLD}\n")
        f.write(f"Evaluation Type: Different-person verification (should be > {THRESHOLD})\n\n")
        
        # Overall metrics
        f.write("="*70 + "\n")
        f.write("OVERALL METRICS\n")
        f.write("="*70 + "\n")
        f.write(f"Total comparison pairs: {overall_metrics['total_pairs']}\n")
        f.write(f"Overall Accuracy: {overall_metrics['overall_accuracy']*100:.2f}%\n")
        f.write(f"Overall FAR: {overall_metrics['overall_far']*100:.2f}%\n")
        f.write(f"Overall FRR: {overall_metrics['overall_frr']*100:.2f}% (N/A for different-person comparisons)\n")
        f.write(f"\nDisparity Analysis:\n")
        f.write(f"  Accuracy disparity: {overall_metrics['accuracy_disparity']*100:.2f}% "
                f"(Best: {overall_metrics['max_accuracy']*100:.2f}%, "
                f"Worst: {overall_metrics['min_accuracy']*100:.2f}%)\n")
        f.write(f"  FAR disparity: {overall_metrics['far_disparity']*100:.2f}% "
                f"(Best: {overall_metrics['min_far']*100:.2f}%, "
                f"Worst: {overall_metrics['max_far']*100:.2f}%)\n\n")
        
        # Per-group metrics
        f.write("="*70 + "\n")
        f.write("PER-GROUP METRICS\n")
        f.write("="*70 + "\n")
        f.write(f"{'Race':<20} {'Pairs':<10} {'Accuracy':<12} {'FAR':<12} {'Mean Dist':<12} {'Std Dist':<12}\n")
        f.write("-"*70 + "\n")
        
        for result in all_results:
            f.write(f"{result['race']:<20} "
                   f"{result['total_pairs']:<10} "
                   f"{result['accuracy']*100:>10.2f}% "
                   f"{result['far']*100:>10.2f}% "
                   f"{result['mean_distance']:>11.4f} "
                   f"{result['std_distance']:>11.4f}\n")
        
        f.write("\n")
        
        # Detailed statistics per group
        f.write("="*70 + "\n")
        f.write("DETAILED STATISTICS PER GROUP\n")
        f.write("="*70 + "\n")
        
        for result in all_results:
            f.write(f"\n{result['race']}:\n")
            f.write(f"  Total images: {result['total_images']}\n")
            f.write(f"  Total pairs: {result['total_pairs']}\n")
            f.write(f"  Correct predictions: {result['correct_predictions']}\n")
            f.write(f"  False acceptances: {result['false_acceptances']}\n")
            f.write(f"  Accuracy: {result['accuracy']*100:.2f}%\n")
            f.write(f"  FAR: {result['far']*100:.2f}%\n")
            f.write(f"  Distance statistics:\n")
            f.write(f"    Mean: {result['mean_distance']:.4f}\n")
            f.write(f"    Std: {result['std_distance']:.4f}\n")
            f.write(f"    Min: {result['min_distance']:.4f}\n")
            f.write(f"    Max: {result['max_distance']:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*70 + "\n")
    
    print(f"\n✓ Report saved to: {output_path}")


def main():
    """
    Main execution function for baseline evaluation.
    """
    print("\n" + "*"*70)
    print("FaceNet Baseline Evaluation on Clean Images")
    print("*"*70)
    print(f"Datasets directory: {DATASETS_DIR}")
    print(f"Threshold: {THRESHOLD} (distance > threshold = different people)\n")
    
    # Load FaceNet model
    print("Loading FaceNet model...")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script directory: {SCRIPT_DIR}")
    
    try:
        model = get_facenet_model()
        if model is None:
            print("\n✗ Error: Could not load FaceNet model.")
            print("\nPlease ensure one of the following:")
            print("1. Install keras-facenet: pip install keras-facenet")
            print("   (This will automatically download the model on first use)")
            print("2. Download FaceNet weights and place in models/ directory")
            print("3. Download frozen graph from David Sandberg repository")
            print("\nModel download link: https://github.com/davidsandberg/facenet")
            print("\nNote: If CreatePerturbedDatasets.py works, the model should be available.")
            print("      Try running that script first to ensure the model is downloaded.")
            return
    except Exception as e:
        import traceback
        print(f"\n✗ Error loading FaceNet model: {e}")
        print("\nFull error traceback:")
        traceback.print_exc()
        print("\nPlease ensure one of the following:")
        print("1. Install keras-facenet: pip install keras-facenet")
        print("   (This will automatically download the model on first use)")
        print("2. Check your internet connection if model needs to be downloaded")
        print("3. Download FaceNet weights manually and place in models/ directory")
        print("\nNote: If CreatePerturbedDatasets.py works, try running it first")
        print("      to ensure the model is downloaded and cached.")
        return
    
    print("✓ FaceNet model loaded successfully\n")
    
    # Evaluate each race dataset
    all_results = []
    
    for race in RACES:
        try:
            result = evaluate_race_dataset(race, model, DATASETS_DIR)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            print(f"✗ Error evaluating {race}: {e}")
            continue
    
    if len(all_results) == 0:
        print("\n✗ Error: No valid results generated. Please check dataset files.")
        return
    
    # Calculate overall metrics
    print("\n" + "="*70)
    print("Calculating overall metrics...")
    print("="*70)
    overall_metrics = calculate_overall_metrics(all_results)
    
    print(f"\nOverall Results:")
    print(f"  Total pairs: {overall_metrics['total_pairs']}")
    print(f"  Overall Accuracy: {overall_metrics['overall_accuracy']*100:.2f}%")
    print(f"  Overall FAR: {overall_metrics['overall_far']*100:.2f}%")
    print(f"  Accuracy disparity: {overall_metrics['accuracy_disparity']*100:.2f}%")
    print(f"  FAR disparity: {overall_metrics['far_disparity']*100:.2f}%")
    
    # Generate report
    report_path = os.path.join(OUTPUT_DIR, 'baseline_evaluation_report.txt')
    generate_report(all_results, overall_metrics, report_path)
    
    print("\n" + "*"*70)
    print("Baseline evaluation complete!")
    print("*"*70)
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

