"""
Process All Malaria Samples - Batch Processing Script
Processes all mzML files and organizes results by group
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.mzml_parser import MZMLParser


def process_all_samples():
    """Process all malaria samples and organize by group."""
    
    print("=" * 70)
    print("Processing All Malaria Samples")
    print("=" * 70)
    print()
    
    # Load metadata
    metadata_path = Path("data/metadata/malaria_metadata.csv")
    if not metadata_path.exists():
        print("❌ Metadata file not found!")
        print(f"Please create: {metadata_path}")
        return
    
    metadata = pd.read_csv(metadata_path)
    print(f"✓ Loaded metadata for {len(metadata)} samples")
    print()
    
    # Initialize parser
    config = {
        'intensity_threshold': 1000.0,
        'snr_threshold': 3.0,
    }
    parser = MZMLParser(config=config)
    
    # Storage for results
    results = {
        'file_info': [],
        'feature_counts': [],
    }
    
    # Process each sample
    print("Processing samples...")
    print("-" * 70)
    
    for idx, row in metadata.iterrows():
        sample_id = row['sample_id']
        group = row['group']
        filename = row['filename']
        file_path = Path("data/raw") / filename
        
        print(f"\n[{idx+1}/{len(metadata)}] {sample_id} ({group})")
        
        if not file_path.exists():
            print(f"  ⚠ File not found: {file_path}")
            continue
        
        # Load file
        success = parser.load_file(str(file_path))
        
        if not success:
            print(f"  ✗ Failed to load")
            continue
        
        # Get file info
        file_info = parser.get_file_info()
        file_info['sample_id'] = sample_id
        file_info['group'] = group
        results['file_info'].append(file_info)
        
        # Extract features
        features = parser.extract_features()
        
        # Save individual feature file
        output_dir = Path("results/processed/individual_features")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{sample_id}_features.csv"
        features.to_csv(output_path, index=False)
        
        # Store summary
        feature_summary = {
            'sample_id': sample_id,
            'group': group,
            'n_features': len(features),
            'rt_min': features['rt'].min() if len(features) > 0 else 0,
            'rt_max': features['rt'].max() if len(features) > 0 else 0,
            'mz_min': features['mz'].min() if len(features) > 0 else 0,
            'mz_max': features['mz'].max() if len(features) > 0 else 0,
            'intensity_mean': features['intensity'].mean() if len(features) > 0 else 0,
            'intensity_median': features['intensity'].median() if len(features) > 0 else 0,
        }
        results['feature_counts'].append(feature_summary)
        
        print(f"  ✓ Extracted {len(features):,} features")
        print(f"  ✓ Saved to: {output_path}")
    
    print()
    print("=" * 70)
    print("Processing Complete!")
    print("=" * 70)
    print()
    
    # Create summary DataFrames
    file_info_df = pd.DataFrame(results['file_info'])
    feature_summary_df = pd.DataFrame(results['feature_counts'])
    
    # Save summaries
    summary_dir = Path("results/processed")
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    file_info_df.to_csv(summary_dir / "file_info_summary.csv", index=False)
    feature_summary_df.to_csv(summary_dir / "feature_counts_summary.csv", index=False)
    
    print(f"✓ File info summary saved to: {summary_dir / 'file_info_summary.csv'}")
    print(f"✓ Feature counts summary saved to: {summary_dir / 'feature_counts_summary.csv'}")
    print()
    
    # Print summary statistics
    print_summary_statistics(feature_summary_df)
    
    # Create visualizations
    create_summary_plots(feature_summary_df, file_info_df)
    
    return feature_summary_df, file_info_df


def print_summary_statistics(feature_summary_df):
    """Print summary statistics by group."""
    
    print("=" * 70)
    print("Summary Statistics by Group")
    print("=" * 70)
    print()
    
    for group in feature_summary_df['group'].unique():
        group_data = feature_summary_df[feature_summary_df['group'] == group]
        
        print(f"{group} Group (n={len(group_data)}):")
        print("-" * 50)
        print(f"  Features per sample:")
        print(f"    Mean:   {group_data['n_features'].mean():,.0f}")
        print(f"    Median: {group_data['n_features'].median():,.0f}")
        print(f"    Range:  {group_data['n_features'].min():,.0f} - {group_data['n_features'].max():,.0f}")
        print()
        print(f"  Retention Time Range:")
        print(f"    Min: {group_data['rt_min'].mean():.2f} - {group_data['rt_max'].mean():.2f} seconds")
        print()
        print(f"  m/z Range:")
        print(f"    Min: {group_data['mz_min'].mean():.2f}")
        print(f"    Max: {group_data['mz_max'].mean():.2f}")
        print()
        print(f"  Mean Intensity:")
        print(f"    {group_data['intensity_mean'].mean():.2e}")
        print()
    
    print("=" * 70)
    print()


def create_summary_plots(feature_summary_df, file_info_df):
    """Create summary visualizations."""
    
    print("Creating summary visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Malaria Dataset - Processing Summary', fontsize=16, fontweight='bold')
    
    # Plot 1: Feature counts by group
    ax1 = axes[0, 0]
    groups = feature_summary_df['group'].unique()
    colors = {'Naive': 'skyblue', 'Semi': 'salmon', 'QC': 'lightgreen'}
    
    for group in groups:
        group_data = feature_summary_df[feature_summary_df['group'] == group]
        ax1.bar(group, group_data['n_features'].mean(), 
                color=colors.get(group, 'gray'),
                alpha=0.7, label=f'{group} (n={len(group_data)})')
        # Add error bars (std)
        ax1.errorbar(group, group_data['n_features'].mean(),
                    yerr=group_data['n_features'].std(),
                    fmt='none', color='black', capsize=5)
    
    ax1.set_ylabel('Average Feature Count', fontsize=11)
    ax1.set_title('Feature Counts by Group', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Feature count distribution
    ax2 = axes[0, 1]
    for group in groups:
        group_data = feature_summary_df[feature_summary_df['group'] == group]
        ax2.scatter([group] * len(group_data), group_data['n_features'],
                   color=colors.get(group, 'gray'), s=100, alpha=0.6)
    
    ax2.set_ylabel('Feature Count', fontsize=11)
    ax2.set_title('Feature Count Distribution', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Total spectra by group
    ax3 = axes[1, 0]
    for group in groups:
        group_data = file_info_df[file_info_df['group'] == group]
        ax3.bar(group, group_data['n_spectra'].mean(),
               color=colors.get(group, 'gray'), alpha=0.7)
    
    ax3.set_ylabel('Average Spectra Count', fontsize=11)
    ax3.set_title('Number of Spectra by Group', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mean intensity comparison
    ax4 = axes[1, 1]
    for group in groups:
        group_data = feature_summary_df[feature_summary_df['group'] == group]
        ax4.bar(group, group_data['intensity_mean'].mean(),
               color=colors.get(group, 'gray'), alpha=0.7)
    
    ax4.set_ylabel('Mean Intensity', fontsize=11)
    ax4.set_title('Average Feature Intensity by Group', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path("results/figures/processing_summary.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Summary plots saved to: {output_path}")
    
    plt.show()


def main():
    """Main function."""
    
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║          Malaria Dataset - Batch Processing                       ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Check if metadata exists
    metadata_path = Path("data/metadata/malaria_metadata.csv")
    if not metadata_path.exists():
        print("⚠ Metadata file not found!")
        print()
        print("Please create the metadata file first:")
        print(f"  {metadata_path}")
        print()
        print("Use the template provided in the artifacts.")
        return
    
    # Process all samples
    feature_summary_df, file_info_df = process_all_samples()
    
    print()
    print("=" * 70)
    print("✓ All samples processed successfully!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. Review the summary statistics above")
    print("2. Check individual feature files in: results/processed/individual_features/")
    print("3. Look at the summary plots in: results/figures/processing_summary.png")
    print("4. Ready to build alignment module to combine samples!")
    print()


if __name__ == "__main__":
    main()