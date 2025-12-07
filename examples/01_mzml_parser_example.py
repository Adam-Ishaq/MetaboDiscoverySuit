"""
Example: Using the mzML Parser with COVID Dataset

This script demonstrates how to use the MZMLParser with your COVID data.
Save this as a Jupyter notebook for interactive exploration.
"""

# %% [markdown]
# # mzML Parser Example - COVID Dataset
# 
# This notebook demonstrates how to parse mzML files and extract features
# for downstream biomarker discovery.

# %% Import libraries
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, str(Path.cwd().parent))

from src.data_processing.mzml_parser import MZMLParser, parse_multiple_files

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# %% [markdown]
# ## Step 1: Initialize the Parser

# %% Initialize parser
# Create parser with custom configuration
config = {
    'intensity_threshold': 1000.0,  # Adjust based on your data
    'snr_threshold': 3.0,
    'mass_error_ppm': 10.0
}

parser = MZMLParser(config=config)
print("✓ Parser initialized")

# %% [markdown]
# ## Step 2: Load a Sample File

# %% Load file
# Replace with your actual file path
mzml_file = "data/raw/covid_sample_001.mzML"

success = parser.load_file(mzml_file)

if success:
    print("✓ File loaded successfully!")
    parser.print_file_summary()
else:
    print("✗ Failed to load file. Check the path.")

# %% [markdown]
# ## Step 3: Explore File Information

# %% Get file info
file_info = parser.get_file_info()

print("File Information:")
print("-" * 50)
for key, value in file_info.items():
    print(f"{key:20s}: {value}")

# %% [markdown]
# ## Step 4: Extract Chromatograms

# %% Extract TIC and BPC
rt_tic, tic = parser.get_tic_chromatogram()
rt_bpc, bpc = parser.get_bpc_chromatogram()

print(f"TIC has {len(tic)} data points")
print(f"BPC has {len(bpc)} data points")

# %% Plot chromatograms
fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# TIC
axes[0].plot(rt_tic / 60, tic, color='blue', linewidth=1)
axes[0].set_ylabel('Total Ion Current', fontsize=12)
axes[0].set_title('Total Ion Chromatogram (TIC)', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# BPC
axes[1].plot(rt_bpc / 60, bpc, color='red', linewidth=1)
axes[1].set_xlabel('Retention Time (minutes)', fontsize=12)
axes[1].set_ylabel('Base Peak Intensity', fontsize=12)
axes[1].set_title('Base Peak Chromatogram (BPC)', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/figures/chromatograms.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 5: Extract and Inspect a Single Spectrum

# %% Extract first MS1 spectrum
spectrum = parser.extract_spectrum(0)

if spectrum:
    print(f"Spectrum Information:")
    print(f"  Scan Index: {spectrum['scan_index']}")
    print(f"  Retention Time: {spectrum['retention_time']:.2f} seconds")
    print(f"  MS Level: {spectrum['ms_level']}")
    print(f"  Number of Peaks: {spectrum['n_peaks']}")
    print(f"  Base Peak m/z: {spectrum['base_peak_mz']:.4f}")
    print(f"  Base Peak Intensity: {spectrum['base_peak_intensity']:.2e}")
    print(f"  Total Ion Current: {spectrum['tic']:.2e}")

# %% Plot spectrum
if spectrum:
    fig, ax = plt.subplots(figsize=(12, 6))
    
    mz = spectrum['mz']
    intensity = spectrum['intensity']
    
    # Plot as vertical lines (traditional mass spectrum style)
    ax.vlines(mz, 0, intensity, color='black', linewidth=0.8)
    
    # Highlight top 10 peaks
    top_10_idx = np.argsort(intensity)[-10:]
    ax.vlines(mz[top_10_idx], 0, intensity[top_10_idx], 
              color='red', linewidth=1.2, label='Top 10 peaks')
    
    ax.set_xlabel('m/z', fontsize=12)
    ax.set_ylabel('Intensity', fontsize=12)
    ax.set_title(f'Mass Spectrum (RT: {spectrum["retention_time"]/60:.2f} min)', 
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/figures/example_spectrum.png', dpi=300, bbox_inches='tight')
    plt.show()

# %% [markdown]
# ## Step 6: Extract All Features

# %% Extract features
print("Extracting features from all MS1 spectra...")
features_df = parser.extract_features(ms_level=1)

print(f"\n✓ Extracted {len(features_df)} features")
print(f"  From {features_df['scan_index'].nunique()} scans")

# Display first few features
print("\nFirst 10 features:")
print(features_df.head(10))

# %% Feature statistics
print("\nFeature Statistics:")
print("-" * 50)
print(f"m/z range:        {features_df['mz'].min():.2f} - {features_df['mz'].max():.2f}")
print(f"RT range:         {features_df['rt'].min()/60:.2f} - {features_df['rt'].max()/60:.2f} minutes")
print(f"Intensity range:  {features_df['intensity'].min():.2e} - {features_df['intensity'].max():.2e}")
print(f"Mean intensity:   {features_df['intensity'].mean():.2e}")
print(f"Median intensity: {features_df['intensity'].median():.2e}")

# %% [markdown]
# ## Step 7: Visualize Feature Distribution

# %% Create feature distribution plots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# m/z distribution
axes[0, 0].hist(features_df['mz'], bins=100, color='skyblue', edgecolor='black')
axes[0, 0].set_xlabel('m/z', fontsize=11)
axes[0, 0].set_ylabel('Frequency', fontsize=11)
axes[0, 0].set_title('m/z Distribution', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# RT distribution
axes[0, 1].hist(features_df['rt']/60, bins=100, color='lightcoral', edgecolor='black')
axes[0, 1].set_xlabel('Retention Time (minutes)', fontsize=11)
axes[0, 1].set_ylabel('Frequency', fontsize=11)
axes[0, 1].set_title('Retention Time Distribution', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Intensity distribution (log scale)
axes[1, 0].hist(np.log10(features_df['intensity']), bins=100, 
                color='lightgreen', edgecolor='black')
axes[1, 0].set_xlabel('log10(Intensity)', fontsize=11)
axes[1, 0].set_ylabel('Frequency', fontsize=11)
axes[1, 0].set_title('Intensity Distribution', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 2D feature map (RT vs m/z)
h = axes[1, 1].hexbin(features_df['rt']/60, features_df['mz'], 
                      C=features_df['intensity'],
                      gridsize=50, cmap='viridis', 
                      reduce_C_function=np.sum, mincnt=1)
axes[1, 1].set_xlabel('Retention Time (minutes)', fontsize=11)
axes[1, 1].set_ylabel('m/z', fontsize=11)
axes[1, 1].set_title('Feature Map (RT vs m/z)', fontsize=12, fontweight='bold')
plt.colorbar(h, ax=axes[1, 1], label='Total Intensity')

plt.tight_layout()
plt.savefig('results/figures/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ## Step 8: Save Features to CSV

# %% Save features
output_path = Path("results/processed") / "covid_sample_001_features.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)

features_df.to_csv(output_path, index=False)
print(f"✓ Features saved to: {output_path}")

# %% [markdown]
# ## Step 9: Process Multiple Files (Optional)
# 
# If you have multiple COVID samples, you can process them all at once.

# %% Process multiple files
# Uncomment and modify paths as needed
"""
mzml_files = [
    "data/raw/covid_positive_001.mzML",
    "data/raw/covid_positive_002.mzML",
    "data/raw/covid_negative_001.mzML",
    "data/raw/covid_negative_002.mzML",
]

results = parse_multiple_files(
    mzml_files,
    config=config,
    output_dir="results/processed/individual_features"
)

# Summary of results
print("\nProcessing Summary:")
print("-" * 50)
for filename, features in results.items():
    print(f"{filename}:")
    print(f"  Features: {len(features)}")
    if len(features) > 0:
        print(f"  m/z range: {features['mz'].min():.2f} - {features['mz'].max():.2f}")
        print(f"  RT range: {features['rt'].min()/60:.2f} - {features['rt'].max()/60:.2f} min")
"""

# %% [markdown]
# ## Next Steps
# 
# After parsing individual files, the next steps are:
# 
# 1. **Align features across samples** - Match peaks between different files
# 2. **Create feature table** - Generate a matrix of samples × features
# 3. **Quality control** - Filter low-quality features
# 4. **Preprocessing** - Normalize and scale data
# 5. **Statistical analysis** - Find significant differences
# 6. **ML-based biomarker discovery** - Use AI to identify biomarkers

# %% [markdown]
# ## Summary
# 
# In this notebook, we:
# - ✅ Loaded an mzML file
# - ✅ Extracted file metadata
# - ✅ Plotted chromatograms (TIC, BPC)
# - ✅ Visualized mass spectra
# - ✅ Extracted all features
# - ✅ Analyzed feature distributions
# - ✅ Saved features to CSV
# 
# The extracted features are now ready for alignment and further analysis!

print("\n" + "="*70)
print("Example completed successfully!")
print("="*70)
