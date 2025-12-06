"""
Data Preparation Script for Demographic Fairness in Adversarial Robustness

This script downloads the FairFace dataset from Hugging Face and creates balanced
demographic datasets organized by race, gender, and age groups.

Dataset Structure:
- 7 races: East_Asian, Indian, Black, White, Middle_Eastern, Latino_Hispanic, Southeast_Asian
- 9 age ranges: 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+
- 2 genders: Male, Female
- Target: ~90 images per race (5 male + 5 female per age range)

Output:
- Saves 7 pickle files (one per race) containing image data and metadata
- Creates race-specific directories with individual image files
- Generates dataset_metadata.txt with summary statistics

Author: Responsible AI Project Team
"""

import os
import random
import pickle
from collections import defaultdict
from datasets import load_dataset
from PIL import Image
import numpy as np

# Setting random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Define Mappings
RACE_MAPPING = {
    0: 'East_Asian',
    1: 'Indian',
    2: 'Black',
    3: 'White',
    4: 'Middle_Eastern',
    5: "Latino_Hispanic",
    6: 'Southeast_Asian'
}

GENDER_MAPPING = {
    0: 'Male',
    1: 'Female'
}

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
# 5 males + 5 females per age range = 10 images
SAMPLES_PER_AGE_GENDER = 5

# 9 age ranges with 10 images from each range
TARGET_IMAGES_PER_RACE = 90

# Output directory: saves datasets in the same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = SCRIPT_DIR


# Downloaded FairFace Dataset
def download_fairface():
    """Downloading Fairface dataset from Hugging Face
    Returns split for testing"""
    print("=" * 70)
    print("STEP 1: Downloading FairFace Dataset")
    print("=" * 70)
    print("This may take a few minutes...")

    # Load dataset from Hugging Face
    dataset = load_dataset("HuggingFaceM4/FairFace", '0.25')

    # Use validation split for testing (10,984 images)
    # Validation split provides a good balance of size and diversity
    val_data = dataset['validation']
    print(f"✓ Dataset downloaded successfully!")
    print(f"  Total validation images: {len(val_data)}")
    print(f"  Each image has: race, gender, age labels")
    print()

    return val_data

# organizing data by race
def organize_by_demographics(val_data):
    """Organize dataset into nested structure:
    organized_data[race][age][gender] = list of images"""

    print("=" * 70)
    print("STEP 2: Organizing Data by Demographics")
    print("=" * 70)

    # Create nested dictionary structure: organized_data[race][age][gender] = list of images
    # defaultdict automatically creates nested dictionaries when accessing new keys
    organized_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for idx, example in enumerate(val_data):
        # Get demographic labels
        race_label = example['race']
        gender_label = example['gender']
        age_label = example['age']

        # Convert labels to readable values from mapping
        race = RACE_MAPPING[race_label]
        gender = GENDER_MAPPING[gender_label]
        age = AGE_MAPPING[age_label]

        # Store image with metadata
        image_data = {
            'index': idx,
            'image': example['image'],
            'race': race,
            'gender': gender,
            'age': age,
            'race_label': race_label,
            'gender_label': gender_label,
            'age_label': age_label
        }

        # Organize by race, then age, then gender
        organized_data[race][age][gender].append(image_data)

    # Print organization summary showing count of images per demographic group
    print("\n✓ Data organized by demographics")
    print("\nAvailable images per demographic group:")
    print("-" * 70)
    print(f"{'Race':<20} {'Age':<10} {'Gender':<10} {'Count'}")
    print("-" * 70)

    for race in RACE_MAPPING.values():
        for age in AGE_MAPPING.values():
            for gender in GENDER_MAPPING.values(): # Use GENDER_MAPPING.values() for consistency
                count = len(organized_data[race][age][gender])
                if count > 0:  # Only print non-empty groups
                    print(f"{race:<20} {age:<10} {gender:<10} {count}")

    print()
    return organized_data


# Create balanced datasets
def create_balanced_dataset(organized_data, samples_per_group=5):
    """
    Create balanced datasets: 90 images per race
    5 males + 5 females per age range (9 age ranges)

    Args:
        organized_data: Nested dict from organize_by_demographics()
        samples_per_group: Number of samples per (race, age, gender) = 5

    Returns:
        list_of_race_datasets: List of dicts, where each dict is a balanced dataset for one race.
    """

    print("=" * 70)
    print("STEP 3: Creating Balanced Dataset")
    print("=" * 70)
    print(f"Target: {samples_per_group} males + {samples_per_group} females per age range")
    print(f"Target per race: {samples_per_group * 2 * len(AGE_MAPPING)} images")
    print()

    # FIX: Changing the return value and logic to create a list of individual race datasets
    list_of_race_datasets = []
    warnings = []

    for race in RACE_MAPPING.values():
        race_samples = []

        for age in AGE_MAPPING.values():
            for gender in GENDER_MAPPING.values():
                # get available data for this combo
                available = organized_data[race][age][gender]

                if len(available) < samples_per_group:
                    # If there aren't enough images, take all available
                    selected = available
                    warnings.append(
                        f"⚠ {race}, {age}, {gender}: only {len(available)}/{samples_per_group} images available"
                    )
                else:
                    # Randomly sample the required number
                    selected = random.sample(available, samples_per_group)

                race_samples.extend(selected)

        # Append the balanced dataset for the current race
        # Storing as a dictionary with the race name as the key and the list of images as the value
        list_of_race_datasets.append({race: race_samples})

    # Print warnings if any
    if warnings:
        print("\n" + "=" * 70)
        print("WARNINGS:")
        print("=" * 70)
        for warning in warnings:
            print(warning)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("BALANCED DATASET SUMMARY (Per Race)")
    print("=" * 70)

    total_images = 0
    for race_data_dict in list_of_race_datasets:
        race = list(race_data_dict.keys())[0] # Extract race name
        images = race_data_dict[race]         # Extract image list

        # Count by gender
        male_count = sum(1 for img in images if img['gender'] == 'Male')
        female_count = sum(1 for img in images if img['gender'] == 'Female')

        # Count by age
        age_counts = defaultdict(int)
        for img in images:
            age_counts[img['age']] += 1

        print(f"\n{race}:")
        print(f"  Total images: {len(images)}")
        print(f"  Gender distribution: {male_count} males, {female_count} females")
        print(f"  Age distribution:")
        for age in AGE_MAPPING.values():
            count = age_counts[age]
            print(f"    {age:<10}: {count} images")

        total_images += len(images)

    print(f"\n{'=' * 70}")
    print(f"TOTAL IMAGES ACROSS ALL RACES: {total_images}")
    print(f"{'=' * 70}\n")

    return list_of_race_datasets

# save dataset
def save_balanced_dataset(list_of_race_datasets, output_dir=OUTPUT_DIR):
    """
        Save balanced datasets to disk

        Saves:
        1. 7 individual Pickle files, one for each race
        2. Individual race subdirectories with images
        """
    print("=" * 70)
    print("STEP 4: Saving Balanced Dataset")
    print("=" * 70)

    # create output directory
    os.makedirs(output_dir, exist_ok=True)
    all_images_total = 0

    # Save individual race datasets as pickle files and images
    for race_data_dict in list_of_race_datasets:
        race = list(race_data_dict.keys())[0] # Extract race name
        images = race_data_dict[race]         # Extract image list

        # Save individual race dataset as pickle
        pickle_path = os.path.join(output_dir, f'{race}.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(race_data_dict, f)
        print(f"✓ Saved race pickle file: {pickle_path}")

        # Save images organized by race
        race_dir = os.path.join(output_dir, race)
        os.makedirs(race_dir, exist_ok=True)

        for idx, img_data in enumerate(images):
            # Create filename with metadata
            filename = f"{race}_{img_data['gender']}_{img_data['age']}_{idx:03d}.jpg"
            filepath = os.path.join(race_dir, filename)

            # Save image
            img_data['image'].save(filepath)

        print(f"✓ Saved {len(images)} images to: {race_dir}/")
        all_images_total += len(images)


    # Save metadata summary (Optional but useful)
    metadata_path = os.path.join(output_dir, 'dataset_metadata.txt')
    with open(metadata_path, 'w') as f:
        f.write("Balanced FairFace Dataset for Adversarial Robustness Study\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Total images: {all_images_total}\n")
        f.write(f"Number of races: {len(list_of_race_datasets)}\n")
        f.write(f"Images per race: ~90 (5 male + 5 female × 9 age ranges)\n\n")

        f.write("Race Breakdown:\n")
        f.write("-" * 70 + "\n")
        for race_data_dict in list_of_race_datasets:
            race = list(race_data_dict.keys())[0]
            images = race_data_dict[race]
            male_count = sum(1 for img in images if img['gender'] == 'Male')
            female_count = sum(1 for img in images if img['gender'] == 'Female')
            f.write(f"{race:<20}: {len(images)} images ({male_count}M / {female_count}F)\n")

    print(f"✓ Saved metadata: {metadata_path}")
    print()

# verifying dataset
def verify_dataset(list_of_race_datasets):
    """
    Verify that balanced dataset meets requirements
    """
    print("=" * 70)
    print("STEP 5: Verifying Dataset")
    print("=" * 70)

    all_checks_passed = True

    # Check 1: Correct number of races
    if len(list_of_race_datasets) != 7:
        print(f"✗ Expected 7 races, got {len(list_of_race_datasets)}")
        all_checks_passed = False
    else:
        print(f"✓ Correct number of races: 7")

    # Check 2: Approximately 90 images per race and Gender balance
    print("\nIndividual race checks:")
    for race_data_dict in list_of_race_datasets:
        race = list(race_data_dict.keys())[0] # Extract race name
        images = race_data_dict[race]         # Extract image list

        # Check 2: Approximately 90 images per race
        if len(images) < 80 or len(images) > 90:
            print(f"⚠ {race}: {len(images)} images (expected ~90)")
            all_checks_passed = False
        else:
            print(f"✓ {race}: {len(images)} images")

        # Check 3: Gender balance
        male_count = sum(1 for img in images if img['gender'] == 'Male')
        female_count = sum(1 for img in images if img['gender'] == 'Female')
        print(f"  Gender: {male_count}M / {female_count}F")

    if all_checks_passed:
        print("\n" + "=" * 70)
        print("✓ ALL CHECKS PASSED - Dataset is ready for experiments!")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("⚠ Some checks failed - review warnings above")
        print("=" * 70)

    print()

# Executing everything
def main():
    """
    Main execution function
    """
    print("\n")
    print("*" * 70)
    print("FAIRFACE DATA PREPARATION FOR ADVERSARIAL ROBUSTNESS STUDY")
    print("*" * 70)
    print(f"Output Directory: {OUTPUT_DIR}")
    print()

    # Step 1: Download dataset
    val_data = download_fairface()

    # Step 2: Organize by demographics
    organized_data = organize_by_demographics(val_data)

    # Step 3: Create balanced dataset (returns a list of 7 race datasets)
    list_of_race_datasets = create_balanced_dataset(
        organized_data,
        samples_per_group=SAMPLES_PER_AGE_GENDER
    )

    # Step 4: Save dataset (saves the 7 individual .pkl files)
    save_balanced_dataset(list_of_race_datasets)

    # Step 5: Verify dataset
    verify_dataset(list_of_race_datasets)

    print("*" * 70)
    print("DATA PREPARATION COMPLETE!")
    print("*" * 70)
    print(f"\nNext steps:")
    print(f"1. Load individual race dataset: pickle.load(open('./{RACE_MAPPING[0]}.pkl', 'rb'))")
    print(f"2. Create test pairs for each race")
    print(f"3. Establish clean baseline with FaceNet")
    print(f"4. Generate adversarial examples")
    print()


if __name__ == "__main__":
    main()

# Results
# ⚠ Middle_Eastern, 0-2, Female: only 2/5 images available
# ⚠ Latino_Hispanic, 70+, Male: only 1/5 images available
# ⚠ Latino_Hispanic, 70+, Female: only 3/5 images available