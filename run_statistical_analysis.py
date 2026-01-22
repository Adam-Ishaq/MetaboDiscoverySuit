"""
Run Statistical Analysis on Preprocessed Malaria Dataset
Performs PCA, differential analysis, and creates visualizations.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.statistical_analysis.statistical_analysis import StatisticalAnalyzer


def create_pca_plot(pca_scores, pca_obj, output_dir="results/figures"):
    """Create PCA visualization."""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define colors for groups
    colors = {'Naive': 'skyblue', 'Semi': 'salmon', 'QC': 'lightgreen'}
    
    # Plot 1: PC1 vs PC2 scatter
    ax1 = axes[0]
    
    for group in pca_scores['group'].unique():
        if group == 'QC':
            continue  # Skip QC for main comparison
        
        group_data = pca_scores[pca_scores['group'] == group]
        ax1.scatter(group_data['PC1'], group_data['PC2'],
                   c=colors[group], s=150, alpha=0.7,
                   edgecolors='black', linewidth=1.5, label=group)
    
    # Plot QC separately
    qc_data = pca_scores[pca_scores['group'] == 'QC']
    if len(qc_data) > 0:
        ax1.scatter(qc_data['PC1'], qc_data['PC2'],
                   c=colors['QC'], s=150, alpha=0.7,
                   marker='s', edgecolors='black', linewidth=1.5, label='QC')
    
    # Add sample labels
    for idx, row in pca_scores.iterrows():
        ax1.annotate(idx, (row['PC1'], row['PC2']),
                    fontsize=8, alpha=0.7,
                    xytext=(5, 5), textcoords='offset points')
    
    var1 = pca_obj.explained_variance_ratio_[0] * 100
    var2 = pca_obj.explained_variance_ratio_[1] * 100
    
    ax1.set_xlabel(f'PC1 ({var1:.1f}% variance)', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'PC2 ({var2:.1f}% variance)', fontsize=12, fontweight='bold')
    ax1.set_title('PCA Score Plot - Naive vs Semi', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    
    # Plot 2: Variance explained
    ax2 = axes[1]
    
    var_explained = pca_obj.explained_variance_ratio_ * 100
    cumsum_var = np.cumsum(var_explained)
    
    x = range(1, len(var_explained) + 1)
    ax2.bar(x, var_explained, alpha=0.7, color='steelblue',
           edgecolor='black', label='Individual')
    ax2.plot(x, cumsum_var, 'ro-', linewidth=2, markersize=8,
            label='Cumulative')
    
    ax2.set_xlabel('Principal Component', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Variance Explained (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Variance Explained by PCs', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for i, (var, cum) in enumerate(zip(var_explained, cumsum_var)):
        ax2.text(i+1, var, f'{var:.1f}%', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "pca_analysis.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ PCA plot saved: {fig_path}")
    
    plt.show()


def create_volcano_plot(diff_results, output_dir="results/figures"):
    """Create volcano plot."""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Prepare data
    x = diff_results['log2_fold_change'].values
    y = -np.log10(diff_results['p_value_adjusted'].values + 1e-300)
    
    # Define significance thresholds
    fc_threshold = 1.0  # log2 fold change
    p_threshold = -np.log10(0.05)
    
    # Color points by significance
    colors = []
    for i, row in diff_results.iterrows():
        if row['p_value_adjusted'] < 0.05 and abs(row['log2_fold_change']) > fc_threshold:
            if row['log2_fold_change'] > 0:
                colors.append('red')  # Upregulated
            else:
                colors.append('blue')  # Downregulated
        else:
            colors.append('gray')  # Not significant
    
    # Scatter plot
    ax.scatter(x, y, c=colors, alpha=0.6, s=20, edgecolors='none')
    
    # Add threshold lines
    ax.axhline(y=p_threshold, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=fc_threshold, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=-fc_threshold, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    # Labels
    ax.set_xlabel('log2(Fold Change) [Semi / Naive]', fontsize=12, fontweight='bold')
    ax.set_ylabel('-log10(Adjusted p-value)', fontsize=12, fontweight='bold')
    ax.set_title('Volcano Plot - Differential Metabolites', fontsize=14, fontweight='bold')
    
    # Legend
    n_up = sum([c == 'red' for c in colors])
    n_down = sum([c == 'blue' for c in colors])
    n_ns = sum([c == 'gray' for c in colors])
    
    legend_elements = [
        Patch(facecolor='red', label=f'Upregulated ({n_up})'),
        Patch(facecolor='blue', label=f'Downregulated ({n_down})'),
        Patch(facecolor='gray', label=f'Not Significant ({n_ns})')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)
    
    # Annotate top features
    top_features = diff_results.nsmallest(5, 'p_value_adjusted')
    for _, row in top_features.iterrows():
        if 'mz' in row and 'rt' in row:
            label = f"m/z {row['mz']:.2f}"
        else:
            label = row['feature_id'][:10]
        
        ax.annotate(label,
                   (row['log2_fold_change'], -np.log10(row['p_value_adjusted'] + 1e-300)),
                   fontsize=8, alpha=0.7,
                   xytext=(5, 5), textcoords='offset points',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    fig_path = output_path / "volcano_plot.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Volcano plot saved: {fig_path}")
    
    plt.show()


def create_heatmap(data, diff_results, metadata, n_features=50, output_dir="results/figures"):
    """Create heatmap of top differential features."""
    
    # Get top features
    top_features = diff_results.nsmallest(n_features, 'p_value_adjusted')
    top_feature_ids = top_features['feature_id'].values
    
    # Get data for these features
    heatmap_data = data[top_feature_ids].T
    
    # Reorder samples by group
    sample_order = []
    for group in ['Naive', 'Semi', 'QC']:
        group_samples = metadata[metadata['group'] == group].index
        sample_order.extend([s for s in group_samples if s in heatmap_data.columns])
    
    heatmap_data = heatmap_data[sample_order]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create heatmap
    sns.heatmap(heatmap_data, cmap='RdBu_r', center=0,
               cbar_kws={'label': 'Scaled Intensity'},
               xticklabels=True, yticklabels=False,
               ax=ax)
    
    # Add group color bar
    group_colors = []
    color_map = {'Naive': 'skyblue', 'Semi': 'salmon', 'QC': 'lightgreen'}
    for sample in heatmap_data.columns:
        if sample in metadata.index:
            group = metadata.loc[sample, 'group']
            group_colors.append(color_map.get(group, 'gray'))
        else:
            group_colors.append('gray')
    
    # Add color bar at top
    for i, color in enumerate(group_colors):
        ax.add_patch(plt.Rectangle((i, -1), 1, 1, facecolor=color, edgecolor='black', linewidth=0.5))
    
    ax.set_xlabel('Samples', fontsize=12, fontweight='bold')
    ax.set_ylabel(f'Top {n_features} Differential Features', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap of Top Differential Metabolites', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    fig_path = output_path / "heatmap_top_features.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Heatmap saved: {fig_path}")
    
    plt.show()


def create_boxplots(data, diff_results, metadata, n_features=6, output_dir="results/figures"):
    """Create box plots for top differential features."""
    
    # Get top features
    top_features = diff_results.nsmallest(n_features, 'p_value_adjusted')
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, (idx, row) in enumerate(top_features.iterrows()):
        if i >= 6:
            break
        
        ax = axes[i]
        feature_id = row['feature_id']
        
        # Get data for this feature
        feature_data = data[feature_id]
        
        # Prepare data for plotting
        plot_data = []
        groups = []
        
        for group in ['Naive', 'Semi']:
            group_samples = metadata[metadata['group'] == group].index
            group_values = feature_data[group_samples]
            plot_data.extend(group_values.values)
            groups.extend([group] * len(group_values))
        
        plot_df = pd.DataFrame({'Group': groups, 'Intensity': plot_data})
        
        # Create box plot
        sns.boxplot(data=plot_df, x='Group', y='Intensity', ax=ax,
                   palette={'Naive': 'skyblue', 'Semi': 'salmon'})
        sns.swarmplot(data=plot_df, x='Group', y='Intensity', ax=ax,
                     color='black', alpha=0.5, size=4)
        
        # Add statistics
        p_val = row['p_value_adjusted']
        fc = row['fold_change']
        
        if 'mz' in row and 'rt' in row:
            title = f"m/z {row['mz']:.2f} @ {row['rt']/60:.2f}min"
        else:
            title = feature_id[:20]
        
        ax.set_title(f"{title}\np={p_val:.2e}, FC={fc:.2f}",
                    fontsize=10, fontweight='bold')
        ax.set_ylabel('Scaled Intensity', fontsize=10)
        ax.set_xlabel('')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Top 6 Differential Metabolites', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    fig_path = output_path / "boxplots_top_features.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Box plots saved: {fig_path}")
    
    plt.show()


def main():
    """Main statistical analysis execution."""
    
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║       Statistical Analysis - Malaria Dataset                      ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Initialize analyzer
    print("Initializing statistical analyzer...")
    analyzer = StatisticalAnalyzer()
    print()
    
    # Load data
    print("=" * 70)
    print("Step 1: Loading Data")
    print("=" * 70)
    analyzer.load_data(
        feature_table_path="results/preprocessed/preprocessed_feature_table.csv",
        sample_metadata_path="data/metadata/malaria_metadata.csv",
        feature_metadata_path="results/preprocessed/preprocessed_feature_metadata.csv"
    )
    print()
    
    # Perform PCA
    print("=" * 70)
    print("Step 2: Principal Component Analysis (PCA)")
    print("=" * 70)
    pca_scores, pca_obj = analyzer.perform_pca(n_components=5)
    print()
    
    # Perform differential analysis
    print("=" * 70)
    print("Step 3: Differential Analysis (Naive vs Semi)")
    print("=" * 70)
    diff_results = analyzer.differential_analysis(
        group_column='group',
        group1='Naive',
        group2='Semi',
        test_type='ttest'
    )
    print()
    
    # Get summary statistics
    print("=" * 70)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 70)
    summary = analyzer.get_summary_statistics()
    
    print(f"\nDataset:")
    print(f"  Total features analyzed:     {summary['total_features']:,}")
    
    print(f"\nSignificance (adjusted p-value):")
    print(f"  Significant (p < 0.05):      {summary['significant_005']:,}")
    print(f"  Significant (p < 0.01):      {summary['significant_001']:,}")
    print(f"  Upregulated in Semi:         {summary['upregulated']:,}")
    print(f"  Downregulated in Semi:       {summary['downregulated']:,}")
    
    print(f"\nEffect Sizes:")
    print(f"  Max fold change:             {summary['max_fold_change']:.2f}x")
    print(f"  Min fold change:             {summary['min_fold_change']:.2f}x")
    print(f"  Minimum p-value:             {summary['min_pvalue']:.2e}")
    
    print(f"\nPCA Results:")
    print(f"  PC1 variance:                {summary['pca_variance_pc1']:.2f}%")
    print(f"  PC2 variance:                {summary['pca_variance_pc2']:.2f}%")
    print(f"  Total variance (5 PCs):      {summary['pca_total_variance']:.2f}%")
    
    print("\n" + "=" * 70)
    print()
    
    # Export results
    print("=" * 70)
    print("Step 4: Exporting Results")
    print("=" * 70)
    analyzer.export_results("results/statistical_analysis")
    print()
    
    # Create visualizations
    print("=" * 70)
    print("Step 5: Creating Visualizations")
    print("=" * 70)
    print("\nCreating PCA plot...")
    create_pca_plot(pca_scores, pca_obj)
    
    print("\nCreating volcano plot...")
    create_volcano_plot(diff_results)
    
    print("\nCreating heatmap...")
    create_heatmap(analyzer.data, diff_results, analyzer.metadata)
    
    print("\nCreating box plots...")
    create_boxplots(analyzer.data, diff_results, analyzer.metadata)
    
    print()
    print("=" * 70)
    print("✓ Statistical Analysis Complete!")
    print("=" * 70)
    print()
    print("Files created:")
    print("  Results:")
    print("    - PCA scores:                    results/statistical_analysis/pca_scores.csv")
    print("    - Differential analysis:         results/statistical_analysis/differential_analysis_results.csv")
    print("    - Significant features:          results/statistical_analysis/significant_features.csv")
    print("    - Top 50 features:               results/statistical_analysis/top50_features.csv")
    print()
    print("  Visualizations:")
    print("    - PCA plot:                      results/figures/pca_analysis.png")
    print("    - Volcano plot:                  results/figures/volcano_plot.png")
    print("    - Heatmap:                       results/figures/heatmap_top_features.png")
    print("    - Box plots:                     results/figures/boxplots_top_features.png")
    print()
    print("Next steps:")
    print("  1. Review the visualizations")
    print("  2. Examine significant_features.csv for biomarker candidates")
    print("  3. Ready for AI/ML model training!")
    print("  4. Phase 6: Machine Learning Biomarker Discovery")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()