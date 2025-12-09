"""
Master Script: Run All Hypothesis Tests

This script runs all hypothesis analysis scripts in the correct order:
1. Generate adversarial data (if not already done)
2. Run H1-H4 hypothesis tests
3. Create final visualizations
4. Generate report

Usage:
    python run_all_hypothesis_tests.py
"""

import os
import sys
import subprocess

# Check if adversarial data exists
from config import ATTACK_TYPES, EPSILONS, get_adversarial_data_path

def check_adversarial_data():
    """Check if adversarial data has been generated"""
    required_files = []
    for attack in ATTACK_TYPES:
        for eps in EPSILONS:
            filename = f'{attack}_e{eps:.2f}_data.csv'
            filepath = get_adversarial_data_path(attack, eps)
            required_files.append((filename, filepath))
    
    missing = [f for f, p in required_files if not os.path.exists(p)]
    
    if missing:
        print(f"\n⚠ Missing {len(missing)} adversarial data files:")
        for f in missing[:5]:  # Show first 5
            print(f"  - {f}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
        return False
    else:
        print("✓ All adversarial data files found")
        return True


def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print(f"\n{'=' * 70}")
    print(f"Running: {description}")
    print(f"{'=' * 70}")
    
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        print(f"✗ Script not found: {script_path}")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(__file__),
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✓ {description} completed successfully")
            return True
        else:
            print(f"✗ {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"✗ Error running {script_name}: {e}")
        return False


def main():
    """Main execution"""
    print("\n" + "*" * 70)
    print("MASTER HYPOTHESIS TESTING SCRIPT")
    print("*" * 70)
    
    # Check if adversarial data exists
    print("\nChecking for adversarial data...")
    has_data = check_adversarial_data()
    
    if not has_data:
        print("\n" + "=" * 70)
        print("STEP 1: Generate Adversarial Data")
        print("=" * 70)
        print("\nAdversarial data not found. Generating now...")
        print("(This will take 3-8 hours)")
        
        response = input("\nProceed with adversarial data generation? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            success = run_script('generate_adv_data.py', 'Adversarial Data Generation')
            if not success:
                print("\n✗ Adversarial data generation failed. Cannot proceed.")
                return
        else:
            print("\nSkipping adversarial data generation.")
            print("Please run generate_adv_data.py manually first.")
            return
    
    # Run hypothesis tests
    print("\n" + "=" * 70)
    print("STEP 2: Running Hypothesis Tests")
    print("=" * 70)
    
    hypothesis_tests = [
        ('H1_Demographic_Disparity/H1_ASR_ANOVA.py', 'H1: Demographic Disparity (ANOVA)'),
        ('H2_Epsilon_Sensitivity/H2_Epsilon_Sensitivity.py', 'H2: Perturbation Budget Sensitivity'),
        ('H3_Compounding_Harm/H3_Compounding_Harm.py', 'H3: Compounding Harm Analysis'),
        ('H4_Attack_Consistency/H4_Attack_Consistency.py', 'H4: Attack Method Consistency'),
    ]
    
    results = {}
    for script, description in hypothesis_tests:
        results[script] = run_script(script, description)
    
    # Create final visualizations
    print("\n" + "=" * 70)
    print("STEP 3: Creating Final Visualizations")
    print("=" * 70)
    
    run_script('create_final_visualizations.py', 'Final Visualization Generation')
    
    # Summary
    print("\n" + "*" * 70)
    print("ALL HYPOTHESIS TESTS COMPLETE!")
    print("*" * 70)
    
    print("\nResults Summary:")
    for script, description in hypothesis_tests:
        status = "✓" if results.get(script, False) else "✗"
        print(f"  {status} {description}")
    
    print("\nOutput Locations:")
    print("  - Adversarial Data: datasets/results/")
    print("  - Hypothesis Results: hypothesis_testing/")
    print("  - Report: hypothesis_testing/analysis_report.md")
    
    print("\nNext Steps:")
    print("  1. Review analysis_report.md")
    print("  2. Check outputs/2_Hypothesis_Stats/ for detailed results")
    print("  3. Review visualizations in outputs/3_Final_Visuals/")


if __name__ == "__main__":
    main()

