"""
Script to create three perturbed versions of each demographic dataset using white-box attacks.

This script requires FaceNet model to compute gradients for FGSM, PGD, and C&W attacks.
Each attack is a white-box attack that needs access to the model's gradients.

Attacks:
- FGSM (Fast Gradient Sign Method): Single-step attack
- PGD (Projected Gradient Descent): Iterative attack with projection
- C&W (Carlini & Wagner): Optimization-based attack

The goal: Create perturbed versions where FaceNet might misidentify two different
people (same race, gender, age) as the same person.
"""

import os
import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
import random
from collections import defaultdict

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

# Perturbation types
PERTURBATION_TYPES = ['FGSM', 'PGD', 'CW']

# Epsilon values to test (perturbation budgets)
EPSILON_VALUES = [0.01, 0.03, 0.05]

# Script directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


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


def preprocess_image(pil_image, target_size=(160, 160)):
    """
    Preprocess PIL image for FaceNet input.
    FaceNet expects images to be 160x160, normalized to [-1, 1] or [0, 1].
    
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


def postprocess_image(tensor_image):
    """
    Convert tensor back to PIL Image.
    
    Args:
        tensor_image: Tensor with shape (1, height, width, 3) or (height, width, 3)
    
    Returns:
        PIL Image
    """
    # Remove batch dimension if present
    if len(tensor_image.shape) == 4:
        tensor_image = tensor_image[0]
    
    # Clamp values to [0, 1] range (ensure valid pixel values)
    tensor_image = tf.clip_by_value(tensor_image, 0.0, 1.0)
    
    # Convert to numpy and scale to [0, 255]
    numpy_array = tensor_image.numpy()
    numpy_array = (numpy_array * 255).astype(np.uint8)
    
    return Image.fromarray(numpy_array)


def fgsm_attack(model, image_tensor, target_embedding, epsilon=0.01):
    """
    Fast Gradient Sign Method (FGSM) Attack
    
    This is a single-step white-box attack that:
    1. Computes the gradient of the loss with respect to the input image
    2. Takes the sign of the gradient (direction only, not magnitude)
    3. Adds epsilon-scaled noise in that direction
    
    Mathematical formula:
        x_adv = x + epsilon * sign(∇_x J(θ, x, y))
    
    Where:
        - x_adv: adversarial (perturbed) image
        - x: original image
        - epsilon: perturbation magnitude (how much noise to add)
        - sign(∇_x J): direction of steepest loss increase
    
    Args:
        model: FaceNet model
        image_tensor: Input image tensor (1, 160, 160, 3)
        target_embedding: Target embedding we want to match (to fool the model)
        epsilon: Perturbation magnitude (typically 0.01-0.1)
    
    Returns:
        Perturbed image tensor
    """
    # Make the image tensor trainable so we can compute gradients
    image_tensor = tf.Variable(image_tensor, trainable=True)
    
    # Use GradientTape to record operations for automatic differentiation
    # This is how we compute gradients in TensorFlow 2.x
    with tf.GradientTape() as tape:
        # Watch the input image so we can compute gradients with respect to it
        tape.watch(image_tensor)
        
        # Get the embedding from FaceNet for the current (potentially perturbed) image
        current_embedding = get_face_embedding(model, image_tensor)
        
        # Calculate loss: we want to minimize the distance between current and target embedding
        # This makes FaceNet think the perturbed image is similar to the target
        # Using MSE (Mean Squared Error) loss
        loss = tf.reduce_mean(tf.square(current_embedding - target_embedding))
    
    # *** CORE LINE: Compute the gradient of loss with respect to the input image ***
    # This tells us: "If I change pixel (i,j) by a small amount, how much does the loss change?"
    gradient = tape.gradient(loss, image_tensor)
    
    # *** CORE LINE: Take the sign of the gradient ***
    # sign() returns +1 for positive gradients, -1 for negative, 0 for zero
    # This gives us the DIRECTION to change each pixel (not the magnitude)
    # We use sign instead of raw gradient to make the attack more robust
    signed_gradient = tf.sign(gradient)
    
    # *** CORE LINE: HERE IS WHERE THE NOISE IS ADDED ***
    # Multiply the signed gradient by epsilon and add it to the original image
    # epsilon controls how much noise we add (larger epsilon = more visible noise)
    # This single step creates the adversarial perturbation
    perturbed_image = image_tensor + epsilon * signed_gradient
    
    # Clip the perturbed image to valid pixel range [0, 1]
    # This ensures the image is still a valid image after perturbation
    perturbed_image = tf.clip_by_value(perturbed_image, 0.0, 1.0)
    
    return perturbed_image


def pgd_attack(model, image_tensor, target_embedding, epsilon=0.01, alpha=0.001, num_iter=10):
    """
    Projected Gradient Descent (PGD) Attack
    
    PGD is an iterative version of FGSM. Instead of one big step, it takes many small steps,
    projecting back to the epsilon-ball after each step.
    
    Mathematical formula (iterative):
        x_0 = x (original image)
        x_{t+1} = Clip_{x, epsilon}(x_t + alpha * sign(∇_x J(θ, x_t, y)))
    
    Where:
        - alpha: step size (smaller than epsilon)
        - num_iter: number of iterations
        - Clip: projects back to epsilon-ball around original image
    
    Args:
        model: FaceNet model
        image_tensor: Input image tensor
        target_embedding: Target embedding to match
        epsilon: Maximum perturbation magnitude (budget)
        alpha: Step size per iteration (typically epsilon/10)
        num_iter: Number of iterations
    
    Returns:
        Perturbed image tensor
    """
    # Store the original image (we'll need it for projection)
    original_image = tf.identity(image_tensor)
    
    # Start with the original image
    perturbed_image = tf.Variable(image_tensor, trainable=True)
    
    # Iterate multiple times (this is what makes PGD stronger than FGSM)
    for iteration in range(num_iter):
        # Make the current perturbed image trainable for gradient computation
        perturbed_image = tf.Variable(perturbed_image, trainable=True)
        
        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            tape.watch(perturbed_image)
            
            # Get embedding for current perturbed image
            current_embedding = get_face_embedding(model, perturbed_image)
            
            # Calculate loss (same as FGSM)
            loss = tf.reduce_mean(tf.square(current_embedding - target_embedding))
        
        # *** CORE LINE: Compute gradient (same as FGSM) ***
        gradient = tape.gradient(loss, perturbed_image)
        
        # *** CORE LINE: Take sign of gradient (same as FGSM) ***
        signed_gradient = tf.sign(gradient)
        
        # *** CORE LINE: HERE IS WHERE A SMALL STEP OF NOISE IS ADDED (per iteration) ***
        # Take a small step (alpha) in the direction of the gradient
        # alpha is much smaller than epsilon, so we take many small steps
        perturbed_image = perturbed_image + alpha * signed_gradient
        
        # Clip to valid pixel range [0, 1]
        perturbed_image = tf.clip_by_value(perturbed_image, 0.0, 1.0)
        
        # *** CORE LINE: PROJECT BACK TO EPSILON-BALL ***
        # Calculate how much we've changed from the original
        delta = perturbed_image - original_image
        
        # Clip the total change to be within epsilon budget
        # This ensures the total perturbation doesn't exceed epsilon
        # This is the "projection" step that makes PGD more controlled than FGSM
        delta = tf.clip_by_value(delta, -epsilon, epsilon)
        
        # Apply the clipped delta to the original image
        # This ensures we stay within the epsilon-ball around the original
        perturbed_image = original_image + delta
        
        # Final clip to ensure valid pixels
        perturbed_image = tf.clip_by_value(perturbed_image, 0.0, 1.0)
    
    return perturbed_image


def cw_attack(model, image_tensor, target_embedding, epsilon=0.01, c=1.0, kappa=0.0,
              learning_rate=0.01, num_iter=100, binary_search_steps=9):
    """
    Carlini & Wagner (C&W) L2 Attack
    
    C&W is an optimization-based attack that finds the minimum perturbation needed.
    It uses a different approach: instead of using sign(gradient), it optimizes
    a variable w that is transformed to create the perturbation.
    
    Mathematical approach:
        1. Transform: x_adv = (tanh(w) + 1) / 2  (keeps values in [0,1])
        2. Optimize: minimize ||x_adv - x||_2^2 + c * max(0, f(x_adv))
        3. Constrain: ||x_adv - x||_∞ <= epsilon (L∞ constraint)
    
    Where:
        - ||x_adv - x||_2^2: L2 norm of perturbation (we want this small)
        - f(x_adv): loss function (0 if attack succeeds, positive otherwise)
        - c: trade-off constant (found via binary search)
        - epsilon: Maximum L∞ perturbation (constraint)
    
    Args:
        model: FaceNet model
        image_tensor: Input image tensor
        target_embedding: Target embedding to match
        epsilon: Maximum L∞ perturbation (for consistency with other attacks)
        c: Initial trade-off constant
        kappa: Confidence parameter (margin for attack success)
        learning_rate: Learning rate for optimizer
        num_iter: Number of optimization iterations
        binary_search_steps: Number of binary search steps to find best c
    
    Returns:
        Perturbed image tensor
    """
    # Store original image
    original_image = image_tensor
    
    # *** CORE LINE: Initialize optimization variable w ***
    # We optimize w (not the image directly) to ensure smooth optimization
    # w will be transformed via tanh to create the perturbation
    # Initialize w such that tanh(w) maps to the original image
    # Inverse of tanh transform: w = atanh(2*x - 1)
    w = tf.Variable(tf.atanh(2.0 * image_tensor - 1.0), trainable=True)
    
    # Set up optimizer (Adam is commonly used for C&W)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    # Binary search for best c value
    c_low = 0.0
    c_high = 1e10
    best_perturbation = None
    best_norm = float('inf')
    
    for binary_step in range(binary_search_steps):
        c = (c_low + c_high) / 2.0
        
        # Reset w for this c value
        w = tf.Variable(tf.atanh(2.0 * image_tensor - 1.0), trainable=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Optimize for num_iter steps
        for iteration in range(num_iter):
            with tf.GradientTape() as tape:
                tape.watch(w)
                
                # *** CORE LINE: Transform w to image space using tanh ***
                # tanh(w) maps to [-1, 1], then we shift and scale to [0, 1]
                # This ensures the image stays in valid range without hard clipping
                # This smooth transformation allows better optimization
                x_adv = (tf.tanh(w) + 1.0) / 2.0
                
                # *** CORE LINE: Apply L∞ constraint (epsilon) for consistency with FGSM/PGD ***
                # Clip perturbation to epsilon budget (L∞ norm constraint)
                # This ensures the perturbation doesn't exceed epsilon in any pixel
                # This makes C&W comparable to FGSM/PGD in terms of perturbation magnitude
                delta = x_adv - original_image
                delta = tf.clip_by_value(delta, -epsilon, epsilon)
                x_adv = original_image + delta
                
                # Get embedding for adversarial image (after epsilon constraint)
                current_embedding = get_face_embedding(model, x_adv)
                
                # Calculate distance in embedding space
                embedding_distance = tf.reduce_mean(tf.square(current_embedding - target_embedding))
                
                # *** CORE LINE: Calculate L2 norm of perturbation ***
                # This measures how much we've changed the image
                # We want this to be as small as possible (within epsilon constraint)
                perturbation_norm = tf.reduce_mean(tf.square(x_adv - original_image))
                
                # *** CORE LINE: Calculate the attack success condition ***
                # f(x_adv) = max(0, embedding_distance - kappa)
                # If embedding_distance < kappa, attack succeeded (f = 0)
                # Otherwise, we penalize the failure
                attack_loss = tf.maximum(0.0, embedding_distance - kappa)
                
                # *** CORE LINE: HERE IS THE TOTAL LOSS WE MINIMIZE ***
                # Total loss = perturbation_size + c * attack_penalty
                # We want: small perturbation AND successful attack
                # c controls the trade-off: larger c = prioritize attack success
                total_loss = perturbation_norm + c * attack_loss
            
            # Compute gradients with respect to w
            gradients = tape.gradient(total_loss, w)
            
            # Update w using the optimizer
            # This is where the optimization happens - w is adjusted to minimize loss
            optimizer.apply_gradients([(gradients, w)])
        
        # Check if attack succeeded (apply epsilon constraint to final result)
        x_adv_final = (tf.tanh(w) + 1.0) / 2.0
        # Apply epsilon constraint to final result
        delta_final = x_adv_final - original_image
        delta_final = tf.clip_by_value(delta_final, -epsilon, epsilon)
        x_adv_final = original_image + delta_final
        final_embedding = get_face_embedding(model, x_adv_final)
        final_distance = tf.reduce_mean(tf.square(final_embedding - target_embedding))
        
        if final_distance < kappa:
            # Attack succeeded - this c value works
            final_norm = tf.reduce_mean(tf.square(x_adv_final - original_image))
            if final_norm < best_norm:
                best_norm = final_norm
                best_perturbation = x_adv_final
            c_high = c  # Try smaller c next time
        else:
            # Attack failed - need larger c
            c_low = c
    
    # Return best perturbation found, or final result if none succeeded
    if best_perturbation is not None:
        return best_perturbation
    else:
        return (tf.tanh(w) + 1.0) / 2.0


def apply_perturbation(image_data, target_image_data, perturbation_type, model, epsilon=0.01):
    """
    Apply a perturbation to a single image to make it similar to a target image.
    
    Args:
        image_data: Dictionary containing image and metadata (source image)
        target_image_data: Dictionary containing target image (what we want to match)
        perturbation_type: Type of perturbation ('CW', 'PGD', 'FGSM')
        model: FaceNet model
        epsilon: Perturbation magnitude (for FGSM and PGD)
    
    Returns:
        Dictionary with perturbed image and same metadata
    """
    # Preprocess both images for FaceNet
    image_tensor = preprocess_image(image_data['image'])
    target_image_tensor = preprocess_image(target_image_data['image'])
    
    # Get target embedding (what we want the perturbed image to match)
    # This is the embedding of a different person (same demographics)
    target_embedding = get_face_embedding(model, target_image_tensor)
    
    # Apply appropriate attack
    if perturbation_type == 'FGSM':
        perturbed_tensor = fgsm_attack(model, image_tensor, target_embedding, epsilon=epsilon)
    elif perturbation_type == 'PGD':
        perturbed_tensor = pgd_attack(model, image_tensor, target_embedding, epsilon=epsilon)
    elif perturbation_type == 'CW':
        perturbed_tensor = cw_attack(model, image_tensor, target_embedding, epsilon=epsilon)
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")
    
    # Convert back to PIL Image
    perturbed_image = postprocess_image(perturbed_tensor)
    
    # Create new image data with perturbed image
    perturbed_data = image_data.copy()
    perturbed_data['image'] = perturbed_image
    perturbed_data['perturbation_type'] = perturbation_type
    perturbed_data['epsilon'] = epsilon  # Store epsilon value used
    perturbed_data['target_index'] = target_image_data['index']  # Track which image was targeted
    
    return perturbed_data


def create_perturbed_dataset(race_name, perturbation_type, model, dataset_dir=OUTPUT_DIR, epsilon=0.01):
    """
    Create a perturbed version of a demographic dataset.
    
    For each image, we:
    1. Find another image from the same gender/age group (different person)
    2. Apply perturbation to make the first image's embedding similar to the second
    3. Goal: FaceNet might misidentify them as the same person
    
    Args:
        race_name: Name of the race
        perturbation_type: Type of perturbation ('FGSM', 'PGD', 'CW')
        model: FaceNet model
        dataset_dir: Directory containing original datasets
        epsilon: Perturbation magnitude
    
    Returns:
        Dictionary with race name and list of perturbed image data
    """
    print(f"\n{'=' * 70}")
    print(f"Creating {perturbation_type} perturbed dataset for {race_name} (ε={epsilon})")
    print(f"{'=' * 70}")
    
    # Load original dataset
    original_dataset = load_dataset(race_name, dataset_dir)
    race_data = original_dataset[race_name]
    
    # Organize by gender and age for pairing
    # We want to pair images from same gender/age but different people
    grouped_data = defaultdict(list)
    for img_data in race_data:
        key = (img_data['gender'], img_data['age'])
        grouped_data[key].append(img_data)
    
    # Create perturbed dataset
    perturbed_data = []
    
    for idx, image_data in enumerate(race_data):
        print(f"Processing image {idx + 1}/{len(race_data)}...", end='\r')
        
        # Find a target image from same gender/age group (different person)
        gender = image_data['gender']
        age = image_data['age']
        same_group = grouped_data[(gender, age)]
        
        # Filter out the current image (we want a different person)
        possible_targets = [img for img in same_group if img['index'] != image_data['index']]
        
        if len(possible_targets) > 0:
            # Randomly select a target from the same group
            target_image_data = random.choice(possible_targets)
        else:
            # Fallback: use a random image from the dataset
            target_image_data = random.choice([img for img in race_data if img['index'] != image_data['index']])
        
        # Apply perturbation to make this image similar to target
        perturbed_image_data = apply_perturbation(
            image_data,
            target_image_data,
            perturbation_type,
            model,
            epsilon=epsilon
        )
        
        perturbed_data.append(perturbed_image_data)
    
    print(f"\n✓ Completed {perturbation_type} perturbation for {race_name}")
    
    # Return in same format as original
    return {race_name: perturbed_data}


def save_perturbed_dataset(perturbed_dataset, race_name, perturbation_type, epsilon, output_dir=OUTPUT_DIR):
    """
    Save perturbed dataset to pickle file in organized folder structure
    
    Folder structure:
    {perturbation_type}/
        {race_name}/
            {race_name}_{epsilon}.pkl
    
    Args:
        perturbed_dataset: Dictionary with race name and perturbed image data
        race_name: Name of the race
        perturbation_type: Type of perturbation ('FGSM', 'PGD', 'CW')
        epsilon: Epsilon value used (for filename)
        output_dir: Base output directory (will create subfolders)
    """
    # Create folder structure: {perturbation_type}/{race_name}/
    attack_folder = os.path.join(output_dir, perturbation_type)
    race_folder = os.path.join(attack_folder, race_name)
    
    # Create directories if they don't exist
    os.makedirs(race_folder, exist_ok=True)
    
    # Create filename with epsilon value (e.g., Black_0.01.pkl)
    filename = f'{race_name}_{epsilon:.2f}.pkl'
    filepath = os.path.join(race_folder, filename)
    
    # Save the dataset
    with open(filepath, 'wb') as f:
        pickle.dump(perturbed_dataset, f)
    
    print(f"✓ Saved: {perturbation_type}/{race_name}/{filename}")


def create_all_perturbed_datasets(model, dataset_dir=OUTPUT_DIR, output_dir=OUTPUT_DIR):
    """
    Create all perturbed versions of all demographic datasets
    
    Creates folder structure:
    {output_dir}/
        FGSM/
            {race}/
                {race}_0.01.pkl
                {race}_0.03.pkl
                {race}_0.05.pkl
        PGD/
            {race}/
                {race}_0.01.pkl
                {race}_0.03.pkl
                {race}_0.05.pkl
        CW/
            {race}/
                {race}_0.01.pkl
                {race}_0.03.pkl
                {race}_0.05.pkl
    
    Args:
        model: FaceNet model (must be loaded)
        dataset_dir: Directory containing original .pkl files
        output_dir: Base directory to save perturbed datasets (will create subfolders)
    """
    print("\n" + "*" * 70)
    print("CREATING PERTURBED DATASETS WITH WHITE-BOX ATTACKS")
    print("*" * 70)
    print(f"Races: {len(RACES)}")
    print(f"Perturbation types: {PERTURBATION_TYPES}")
    print(f"Epsilon values: {EPSILON_VALUES}")
    print(f"Total datasets to create: {len(RACES) * len(PERTURBATION_TYPES) * len(EPSILON_VALUES)}")
    print()
    print("Folder structure:")
    print("  {attack_type}/")
    print("    {race}/")
    print("      {race}_{epsilon}.pkl")
    print()
    
    total_created = 0
    
    # Iterate through each attack type
    for perturbation_type in PERTURBATION_TYPES:
        print(f"\n{'#' * 70}")
        print(f"PROCESSING {perturbation_type} ATTACKS")
        print(f"{'#' * 70}")
        
        # For C&W, epsilon is handled differently (it uses optimization)
        # But we'll still create versions for consistency
        epsilons_to_use = EPSILON_VALUES if perturbation_type != 'CW' else EPSILON_VALUES
        
        # Iterate through each race
        for race in RACES:
            print(f"\n--- Processing {race} ---")
            
            # Iterate through each epsilon value
            for epsilon in epsilons_to_use:
                try:
                    # Create perturbed dataset with this epsilon
                    perturbed_dataset = create_perturbed_dataset(
                        race,
                        perturbation_type,
                        model,
                        dataset_dir,
                        epsilon=epsilon
                    )
                    
                    # Save perturbed dataset in organized folder structure
                    save_perturbed_dataset(
                        perturbed_dataset,
                        race,
                        perturbation_type,
                        epsilon,
                        output_dir
                    )
                    
                    total_created += 1
                    
                except Exception as e:
                    print(f"✗ Error processing {race} with {perturbation_type} (ε={epsilon}): {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
    
    print("\n" + "*" * 70)
    print("PERTURBED DATASET CREATION COMPLETE!")
    print("*" * 70)
    print(f"\nCreated {total_created} perturbed datasets")
    print(f"  - {len(PERTURBATION_TYPES)} attack types")
    print(f"  - {len(RACES)} races per attack")
    print(f"  - {len(EPSILON_VALUES)} epsilon values per race")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    for attack_type in PERTURBATION_TYPES:
        print(f"    {attack_type}/")
        print(f"      {{race}}/")
        for eps in EPSILON_VALUES:
            print(f"        {{race}}_{eps:.2f}.pkl")
    print("\nNext steps:")
    print("1. Create benchmark pairs (same race, gender, age range)")
    print("2. Test with FaceNet to verify baseline (faces should be identified as different)")
    print("3. Test again with perturbed versions at different epsilon values")
    print("4. Analyze demographic disparities across perturbation magnitudes")
    print()


def main():
    """
    Main execution function
    """
    print("\n" + "*" * 70)
    print("PERTURBED DATASET CREATION SCRIPT")
    print("Using White-Box Attacks (FGSM, PGD, C&W)")
    print("*" * 70)
    print("\nThis script creates perturbed versions using FaceNet gradients.")
    print("FaceNet model is required for computing attack gradients.")
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
    
    # Create all perturbed datasets
    # This will create datasets for all epsilon values: [0.01, 0.03, 0.05]
    # Organized in folder structure: {attack_type}/{race}/{race}_{epsilon}.pkl
    create_all_perturbed_datasets(model)


if __name__ == "__main__":
    main()

