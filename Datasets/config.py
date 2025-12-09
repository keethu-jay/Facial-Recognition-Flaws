"""
Configuration file for adversarial robustness study
Stores all key parameters for reproducibility
"""

# Target Model Parameters
DISTANCE_THRESHOLD = 1.0  # FaceNet verification threshold (tau)

# Adversarial Attack Parameters
ATTACK_TYPES = ['FGSM', 'PGD', 'CW']
EPSILONS = [0.01, 0.03, 0.05]
PGD_STEPS = 10  # Number of iterations for PGD attack

# Visualization Parameters
COLOR_PALETTE = 'coolwarm_r'  # Red/Green/Blue divergent palette (reversed for Red=High Error)
SIGNIFICANCE_ALPHA = 0.05

# Race color mapping (for consistency)
RACE_COLORS = {
    'East_Asian': '#FF6B6B',
    'Indian': '#4ECDC4',
    'Black': '#45B7D1',
    'White': '#FFA07A',
    'Middle_Eastern': '#98D8C8',
    'Latino_Hispanic': '#F7DC6F',
    'Southeast_Asian': '#BB8FCE'
}

# Race definitions
RACES = [
    'East_Asian',
    'Indian',
    'Black',
    'White',
    'Middle_Eastern',
    'Latino_Hispanic',
    'Southeast_Asian'
]

# Age groups
AGE_GROUPS = [
    '0-2',
    '3-9',
    '10-19',
    '20-29',
    '30-39',
    '40-49',
    '50-59',
    '60-69',
    '70+'
]

# Gender
GENDERS = ['Male', 'Female']

# File paths
BASELINE_DIR = 'datasets/baseline_stats'
OUTPUT_DIR = 'datasets/results'
ADVERSARIAL_DATA_DIR = 'datasets/results'
HYPOTHESIS_STATS_DIR = 'hypothesis_testing'
FINAL_VISUALS_DIR = 'hypothesis_testing'

def get_adversarial_data_path(attack_type, epsilon):
    """Get path to adversarial data CSV for specific attack and epsilon"""
    import os
    filename = f'{attack_type}_e{epsilon:.2f}_data.csv'
    return os.path.join(ADVERSARIAL_DATA_DIR, attack_type, f'e{epsilon:.2f}', filename)

