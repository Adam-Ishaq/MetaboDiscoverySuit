"""
Run Feature Alignment on Malaria Dataset
Aligns features across all samples and creates feature table.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_processing.alignment import FeatureAligner


def visualize_alignment_results(feature_table, feature_metadata, output_dir="results/figures"):
    """Create visualizations of alignment results."""
    
    print("\nCreating visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Feature Alignment Results - Malaria Dataset', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Feature detection frequency
    ax1 = axes[0, 0]
    detection_freq = (feature_table > 0).sum(axis=0)
    ax1.hist(detection_freq, bins=30, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Number of Samples', fontsize=11)
    ax1.set_ylabel('Number of Features', fontsize=11)
    ax1.set_title('Feature Detection Frequency', fontsize=12, fontweight='bold')
    ax1.axvline(detection_freq.median(), color='red', linestyle='--', 
                label=f'Median: {detection_freq.median():.0f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: m/z distribution of aligned features
    ax2 = axes[0, 1]
    ax2.hist(feature_metadata['mz'], bins=50, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('m/z', fontsize=11)
    ax2.set_ylabel('Number of Features', fontsize=11)
    ax2.set_title('m/z Distribution of Aligned Features', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: RT distribution of aligned features
    ax3 = axes[0, 2]
    ax3.hist(feature_metadata['rt']/60, bins=50, color='lightgreen', edgecolor='black')
    ax3.set_xlabel('Retention Time (minutes)', fontsize=11)
    ax3.set_ylabel('Number of Features', fontsize=11)
    ax3.set_title('RT Distribution of Aligned Features', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Missing values heatmap
    ax4 = axes[1, 0]
    missing_matrix = (feature_table == 0).astype(int)
    # Sample 100 features for visualization
    if len(feature_table.columns) > 100:
        sample_cols = np.random.choice(feature_table.columns, 100, replace=False)
        missing_sample = missing_matrix[sample_cols]
    else:
        missing_sample = missing_matrix
    
    sns.heatmap(missing_sample.T, cmap='RdYlGn_r', cbar_kws={'label': 'Missing'},
                ax=ax4, yticklabels=False)
    ax4.set_xlabel('Sample', fontsize=11)
    ax4.set_ylabel('Features (sample)', fontsize=11)
    ax4.set_title('Missing Value Pattern', fontsize=12, fontweight='bold')
    
    # Plot 5: Feature intensity distribution
    ax5 = axes[1, 1]
    # Log scale for better visualization
    feature_means = feature_table.mean(axis=0)
    feature_means = feature_means[feature_means > 0]
    ax5.hist(np.log10(feature_means), bins=50, color='plum', edgecolor='black')
    ax5.set_xlabel('log10(Mean Intensity)', fontsize=11)
    ax5.set_ylabel('Number of Features', fontsize=11)
    ax5.set_title('Feature Intensity Distribution', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Features per sample
    ax6 = axes[1, 2]
    features_per_sample = (feature_table > 0).sum(axis=1)
    
    # Color by group if metadata available
    metadata_path = Path("data/metadata/malaria_metadata.csv")
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        # Match sample order
        sample_groups = []
        for sample in feature_table.index:
            group = metadata[metadata['sample_id'] == sample]['group'].values
            sample_groups.append(group[0] if len(group) > 0 else 'Unknown')
        
        colors = {'Naive': 'skyblue', 'Semi': 'salmon', 'QC': 'lightgreen'}
        bar_colors = [colors.get(g, 'gray') for g in sample_groups]
        
        ax6.bar(range(len(features_per_sample)), features_per_sample, 
               color=bar_colors, edgecolor='black')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[g], label=g) 
                          for g in ['Naive', 'Semi', 'QC'] if g in sample_groups]
        ax6.legend(handles=legend_elements)
    else:
        ax6.bar(range(len(features_per_sample)), features_per_sample, 
               color='steelblue', edgecolor='black')
    
    ax6.set_xlabel('Sample Index', fontsize=11)
    ax6.set_ylabel('Number of Features', fontsize=11)
    ax6.set_title('Features per Sample', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "alignment_results.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Alignment visualizations saved: {fig_path}")
    
    plt.show()


def print_alignment_summary(feature_table, feature_metadata):
    """Print detailed alignment summary."""
    
    print("\n" + "=" * 70)
    print("FEATURE ALIGNMENT SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Dataset Dimensions:':<30}")
    print(f"  Samples:                     {feature_table.shape[0]}")
    print(f"  Aligned Features:            {feature_table.shape[1]:,}")
    
    print(f"\n{'Feature Characteristics:':<30}")
    print(f"  m/z range:                   {feature_metadata['mz'].min():.2f} - {feature_metadata['mz'].max():.2f}")
    print(f"  RT range:                    {feature_metadata['rt'].min()/60:.2f} - {feature_metadata['rt'].max()/60:.2f} minutes")
    
    print(f"\n{'Detection Statistics:':<30}")
    detection_per_feature = (feature_table > 0).sum(axis=0)
    print(f"  Mean detection per feature:  {detection_per_feature.mean():.1f} samples")
    print(f"  Median detection:            {detection_per_feature.median():.0f} samples")
    print(f"  Features in all samples:     {(detection_per_feature == feature_table.shape[0]).sum()}")
    print(f"  Features in ≥50% samples:    {(detection_per_feature >= feature_table.shape[0]/2).sum()}")
    
    print(f"\n{'Missing Values:':<30}")
    total_values = feature_table.size
    missing_values = (feature_table == 0).sum().sum()
    print(f"  Total values:                {total_values:,}")
    print(f"  Missing (zero) values:       {missing_values:,} ({missing_values/total_values*100:.1f}%)")
    print(f"  Present values:              {total_values - missing_values:,} ({(total_values-missing_values)/total_values*100:.1f}%)")
    
    print(f"\n{'Intensity Statistics:':<30}")
    non_zero = feature_table[feature_table > 0]
    print(f"  Mean intensity:              {non_zero.mean().mean():.2e}")
    print(f"  Median intensity:            {non_zero.median().median():.2e}")
    print(f"  Intensity range:             {non_zero.min().min():.2e} - {non_zero.max().max():.2e}")
    
    print(f"\n{'Features per Sample:':<30}")
    features_per_sample = (feature_table > 0).sum(axis=1)
    print(f"  Mean:                        {features_per_sample.mean():.0f}")
    print(f"  Min:                         {features_per_sample.min()}")
    print(f"  Max:                         {features_per_sample.max()}")
    
    print("\n" + "=" * 70)


def main():
    """Main alignment execution."""
    
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║            Feature Alignment - Malaria Dataset                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Configuration
    config = {
        'feature_dir': "results/processed/individual_features",
        'metadata_path': "data/metadata/malaria_metadata.csv",
        'output_dir': "results/aligned",
        'mz_tolerance': 0.01,      # 0.01 Da
        'rt_tolerance': 30.0,      # 30 seconds
        'min_fraction': 0.3,       # Feature must be in ≥30% of samples
    }
    
    print("Configuration:")
    print("-" * 70)
    for key, value in config.items():
        print(f"  {key:<25}: {value}")
    print()
    
    # Initialize aligner
    print("Initializing aligner...")
    aligner = FeatureAligner(
        mz_tolerance=config['mz_tolerance'],
        rt_tolerance=config['rt_tolerance'],
        min_fraction=config['min_fraction']
    )
    
    # Load feature files
    print("\nLoading feature files...")
    n_files = aligner.load_feature_files(
        config['feature_dir'],
        config['metadata_path']
    )
    
    if n_files == 0:
        print("❌ No feature files found. Make sure you ran process_all_malaria_samples.py first!")
        return
    
    print(f"✓ Loaded {n_files} feature files")
    print()
    
    # Align features
    print("Performing alignment...")
    print("(This may take a few minutes for large datasets)")
    print()
    
    aligned_features = aligner.align_features()
    
    if len(aligned_features) == 0:
        print("❌ Alignment failed. Check logs for errors.")
        return
    
    # Create feature table
    print("\nCreating feature table...")
    feature_table = aligner.create_feature_table(fill_missing='zero')
    
    # Get feature metadata
    feature_metadata = aligner.get_feature_metadata()
    
    # Save results
    print("\nSaving results...")
    aligner.save_results(config['output_dir'])
    
    # Print statistics
    stats = aligner.get_alignment_stats()
    
    print("\n" + "=" * 70)
    print("ALIGNMENT STATISTICS")
    print("=" * 70)
    print(f"Samples:                    {stats['n_samples']}")
    print(f"Features before alignment:  {stats['n_total_features_before']:,}")
    print(f"Features after alignment:   {stats['n_features']:,}")
    print(f"Compression ratio:          {stats['compression_ratio']:.1f}x")
    print(f"Missing values:             {stats['missing_values']:,} ({stats['missing_percentage']:.1f}%)")
    print(f"Avg features per sample:    {stats['mean_features_per_sample']:.0f}")
    print("=" * 70)
    
    # Print detailed summary
    print_alignment_summary(feature_table, feature_metadata)
    
    # Create visualizations
    visualize_alignment_results(feature_table, feature_metadata)
    
    print("\n" + "=" * 70)
    print("✓ Alignment Complete!")
    print("=" * 70)
    print()
    print("Files created:")
    print(f"  - Feature table:           results/aligned/feature_table.csv")
    print(f"  - Transposed table:        results/aligned/feature_table_transposed.csv")
    print(f"  - Feature metadata:        results/aligned/feature_metadata.csv")
    print(f"  - Visualizations:          results/figures/alignment_results.png")
    print()
    print("Next steps:")
    print("  1. Review the alignment visualizations")
    print("  2. Check feature_table.csv (samples × features)")
    print("  3. Ready for preprocessing and statistical analysis!")
    print("  4. Then: AI/ML biomarker discovery!")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()