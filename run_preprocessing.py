"""
Run Preprocessing on Aligned Malaria Dataset
Applies quality filtering, normalization, transformation, and scaling.
"""

import sys
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.preprocessing.preprocessing import MetabolomicsPreprocessor


def visualize_preprocessing_effects(original_data, processed_data, output_dir="results/figures"):
    """Create before/after preprocessing visualizations."""
    
    print("\nCreating preprocessing visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Preprocessing Effects - Malaria Dataset', fontsize=16, fontweight='bold')
    
    # Plot 1: Number of features (before/after)
    ax1 = axes[0, 0]
    bars = ax1.bar(['Before', 'After'], 
                   [original_data.shape[1], processed_data.shape[1]],
                   color=['lightcoral', 'lightgreen'], edgecolor='black')
    ax1.set_ylabel('Number of Features', fontsize=11)
    ax1.set_title('Feature Count', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height):,}',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Missing values (before/after)
    ax2 = axes[0, 1]
    missing_before = (original_data == 0).sum().sum() / original_data.size * 100
    missing_after = (processed_data == 0).sum().sum() / processed_data.size * 100
    bars = ax2.bar(['Before', 'After'], [missing_before, missing_after],
                   color=['salmon', 'palegreen'], edgecolor='black')
    ax2.set_ylabel('Missing Values (%)', fontsize=11)
    ax2.set_title('Missing Value Percentage', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Distribution comparison (before)
    ax3 = axes[0, 2]
    original_nonzero = original_data[original_data > 0].values.flatten()
    original_nonzero = original_nonzero[~np.isnan(original_nonzero)]
    if len(original_nonzero) > 0:
        ax3.hist(np.log10(original_nonzero), bins=50, color='skyblue', 
                edgecolor='black', alpha=0.7)
    ax3.set_xlabel('log10(Intensity)', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Original Data Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Distribution comparison (after)
    ax4 = axes[1, 0]
    processed_nonzero = processed_data[processed_data > 0].values.flatten()
    processed_nonzero = processed_nonzero[~np.isnan(processed_nonzero)]
    if len(processed_nonzero) > 0:
        ax4.hist(processed_nonzero, bins=50, color='lightgreen', 
                edgecolor='black', alpha=0.7)
    ax4.set_xlabel('Transformed Intensity', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Processed Data Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Sample normalization effect
    ax5 = axes[1, 1]
    sample_sums_before = original_data.sum(axis=1)
    sample_sums_after = processed_data.sum(axis=1)
    
    x = np.arange(len(sample_sums_before))
    width = 0.35
    ax5.bar(x - width/2, sample_sums_before / sample_sums_before.median(), 
           width, label='Before', color='lightcoral', edgecolor='black', alpha=0.7)
    ax5.bar(x + width/2, sample_sums_after / sample_sums_after.median(), 
           width, label='After', color='lightgreen', edgecolor='black', alpha=0.7)
    
    ax5.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='Target')
    ax5.set_xlabel('Sample Index', fontsize=11)
    ax5.set_ylabel('Relative Total Intensity', fontsize=11)
    ax5.set_title('Normalization Effect', fontsize=12, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: CV comparison (before/after)
    ax6 = axes[1, 2]
    
    # Calculate CV for each feature (before)
    cv_before = (original_data.replace(0, np.nan).std() / 
                original_data.replace(0, np.nan).mean() * 100)
    cv_before = cv_before[~np.isnan(cv_before)]
    
    # Calculate CV for each feature (after)
    cv_after = (processed_data.replace(0, np.nan).std() / 
               processed_data.replace(0, np.nan).mean() * 100)
    cv_after = cv_after[~np.isnan(cv_after)]
    
    ax6.boxplot([cv_before, cv_after], labels=['Before', 'After'],
               patch_artist=True,
               boxprops=dict(facecolor='lightblue', alpha=0.7),
               medianprops=dict(color='red', linewidth=2))
    ax6.set_ylabel('Coefficient of Variation (%)', fontsize=11)
    ax6.set_title('Feature Variability', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "preprocessing_effects.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Preprocessing visualizations saved: {fig_path}")
    
    plt.show()


def main():
    """Main preprocessing execution."""
    
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║          Data Preprocessing - Malaria Dataset                     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Configuration
    config = {
        'feature_table': 'results/aligned/feature_table.csv',
        'feature_metadata': 'results/aligned/feature_metadata.csv',
        'sample_metadata': 'data/metadata/malaria_metadata.csv',
        'output_dir': 'results/preprocessed',
        
        # Filtering parameters
        'max_missing_percent': 50.0,       # Remove features with >50% missing
        'min_intensity': 5000.0,           # Minimum mean intensity
        'min_detection_rate': 0.3,         # Must be in ≥30% of samples
        'max_rsd_qc': 30.0,                # Max RSD in QC samples
        
        # Preprocessing methods
        'normalization': 'median',         # 'sum', 'median', 'quantile', 'none'
        'transformation': 'log2',          # 'log2', 'log10', 'sqrt', 'none'
        'scaling': 'pareto',               # 'auto', 'pareto', 'range', 'none'
        'imputation': 'min',               # 'knn', 'min', 'median', 'zero'
    }
    
    print("Configuration:")
    print("-" * 70)
    print("\nFiltering:")
    print(f"  Max missing values:         {config['max_missing_percent']}%")
    print(f"  Min intensity:              {config['min_intensity']}")
    print(f"  Min detection rate:         {config['min_detection_rate']*100}%")
    print(f"  Max RSD (QC):               {config['max_rsd_qc']}%")
    print("\nPreprocessing:")
    print(f"  Normalization:              {config['normalization']}")
    print(f"  Transformation:             {config['transformation']}")
    print(f"  Scaling:                    {config['scaling']}")
    print(f"  Missing value imputation:   {config['imputation']}")
    print()
    
    # Initialize preprocessor
    print("Initializing preprocessor...")
    preprocessor = MetabolomicsPreprocessor()
    print()
    
    # Load data
    print("=" * 70)
    print("Step 1: Loading Data")
    print("=" * 70)
    preprocessor.load_data(
        feature_table_path=config['feature_table'],
        feature_metadata_path=config['feature_metadata'],
        sample_metadata_path=config['sample_metadata']
    )
    
    # Store original for comparison
    original_data = preprocessor.original_data.copy()
    
    print()
    
    # Step 2: Quality filtering
    print("=" * 70)
    print("Step 2: Quality Filtering")
    print("=" * 70)
    
    preprocessor.filter_missing_values(max_missing_percent=config['max_missing_percent'])
    preprocessor.filter_low_intensity(min_intensity=config['min_intensity'])
    preprocessor.filter_by_detection_rate(min_detection_rate=config['min_detection_rate'])
    preprocessor.filter_by_rsd(max_rsd=config['max_rsd_qc'], use_qc=True)
    
    print()
    
    # Step 3: Normalization
    print("=" * 70)
    print("Step 3: Normalization")
    print("=" * 70)
    preprocessor.normalize(method=config['normalization'])
    print()
    
    # Step 4: Transformation
    print("=" * 70)
    print("Step 4: Transformation")
    print("=" * 70)
    preprocessor.transform(method=config['transformation'])
    print()
    
    # Step 5: Scaling
    print("=" * 70)
    print("Step 5: Scaling")
    print("=" * 70)
    preprocessor.scale(method=config['scaling'])
    print()
    
    # Step 6: Imputation
    print("=" * 70)
    print("Step 6: Missing Value Imputation")
    print("=" * 70)
    preprocessor.impute_missing_values(method=config['imputation'])
    print()
    
    # Print summary
    print(preprocessor.get_processing_summary())
    print()
    
    # Save results
    print("=" * 70)
    print("Saving Results")
    print("=" * 70)
    preprocessor.save_processed_data(config['output_dir'])
    print()
    
    # Create visualizations
    visualize_preprocessing_effects(original_data, preprocessor.processed_data)
    
    print()
    print("=" * 70)
    print("✓ Preprocessing Complete!")
    print("=" * 70)
    print()
    print("Files created:")
    print(f"  - Preprocessed table:      {config['output_dir']}/preprocessed_feature_table.csv")
    print(f"  - Transposed table:        {config['output_dir']}/preprocessed_feature_table_transposed.csv")
    print(f"  - Feature metadata:        {config['output_dir']}/preprocessed_feature_metadata.csv")
    print(f"  - Processing log:          {config['output_dir']}/preprocessing_log.txt")
    print(f"  - Visualizations:          results/figures/preprocessing_effects.png")
    print()
    print("Next steps:")
    print("  1. Review the preprocessing effects visualization")
    print("  2. Check the processing log for details")
    print("  3. Ready for statistical analysis!")
    print("  4. PCA, t-tests, volcano plots coming next!")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()