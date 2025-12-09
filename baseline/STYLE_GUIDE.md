# Visualization Style Guide

This document defines the color scheme and styling conventions used in all baseline visualizations.

## Color Palette

### Race Colors
Each race is assigned a consistent color across all visualizations:

- **East_Asian**: `#FF6B6B` (Red)
- **Indian**: `#4ECDC4` (Teal)
- **Black**: `#45B7D1` (Blue)
- **White**: `#FFA07A` (Light Salmon)
- **Middle_Eastern**: `#98D8C8` (Mint)
- **Latino_Hispanic**: `#F7DC6F` (Yellow)
- **Southeast_Asian**: `#BB8FCE` (Purple)

### Gender Colors
- **Male**: `#3498db` (Blue)
- **Female**: `#e74c3c` (Red)

## Visualization Conventions

### Markers
- **Circles (○)**: Female images
- **Triangles (△)**: Male images

### Plot Styles
- All plots use `whitegrid` style from seaborn
- Figure size: 12x8 inches (or adjusted for subplots)
- DPI: 300 for publication quality
- Font sizes:
  - Title: 14-16pt, bold
  - Axis labels: 12pt, bold
  - Legend: 8-10pt
  - Value labels: 10pt, bold

### File Naming
All visualization files are saved in the `baseline/` directory with descriptive names:
- `fsr_by_gender_per_race.png` - Bar plot comparing FSR by gender for each race
- `male_vs_female_fsr.png` - Overall male vs female comparison
- `total_fsr_by_race.png` - Average FSR by race
- `gender_breakup_pie_charts.png` - Pie charts showing gender distribution for each race
- `fsr_scatterplot_by_image.png` - Scatterplot of FSR for each individual image

## Data Files

### CSV Files
- `{race}_baseline.csv` - Individual race statistics (one per race)
- `{race}_pairs.csv` - Pair information for reuse with perturbed images (one per race)
- `all_races_baseline.csv` - Combined results from all races
- `all_races_pairs.csv` - Combined pair information from all races
- `summary_statistics.csv` - Aggregated summary statistics

### Pickle Files
- `{race}_pairs.pkl` - Pair information in pickle format (faster loading)
- `all_races_pairs.pkl` - Combined pair information in pickle format

### Image-Level CSV Columns (`{race}_baseline.csv`)
- `race`: Race name
- `index`: Original image index
- `age`: Age range
- `gender`: Gender (Male/Female)
- `false_accepts`: Number of false accepts for this image
- `total_pairs`: Total number of pairs this image was tested in
- `fsr`: False Acceptance Rate (false_accepts / total_pairs)
- `distance_threshold`: Threshold used (typically 1.0)

### Pair-Level CSV Columns (`{race}_pairs.csv`)
- `pair_id`: Unique identifier for each pair (format: `{race}_{counter:06d}`)
- `race`: Race name
- `age`: Age range
- `gender`: Gender (Male/Female)
- `image_A_index`: Index of first image in pair
- `image_B_index`: Index of second image in pair
- `image_A_age`: Age of first image
- `image_A_gender`: Gender of first image
- `image_B_age`: Age of second image
- `image_B_gender`: Gender of second image
- `baseline_distance`: Distance calculated on original images
- `baseline_is_false_accept`: Whether this pair was a false accept in baseline
- `distance_threshold`: Threshold used (typically 1.0)

## Using Pairs for Perturbed Image Testing

The pair files are essential for comparing baseline vs perturbed results:

1. **Load pairs**: Use `{race}_pairs.csv` or `{race}_pairs.pkl` to get the exact pairs tested
2. **Reuse same pairs**: When testing perturbed images, use the same `image_A_index` and `image_B_index` pairs
3. **Compare results**: Compare `baseline_distance` vs perturbed distance to measure attack effectiveness
4. **Track changes**: Use `pair_id` to track how each specific pair's distance changes after perturbation

Example workflow:
```python
# Load baseline pairs
pairs_df = pd.read_csv('baseline/Black_pairs.csv')

# For each pair, test with perturbed images
for _, pair in pairs_df.iterrows():
    img_A_perturbed = load_perturbed_image(pair['image_A_index'], 'FGSM', 0.01)
    img_B_original = load_original_image(pair['image_B_index'])
    
    # Calculate distance and compare to baseline
    perturbed_distance = calculate_distance(img_A_perturbed, img_B_original)
    baseline_distance = pair['baseline_distance']
    
    # Track the change
    print(f"Pair {pair['pair_id']}: {baseline_distance:.4f} -> {perturbed_distance:.4f}")
```

