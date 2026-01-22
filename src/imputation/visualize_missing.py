import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ProgenesisDataLoader:
    """
    Loads and parses Progenesis QI normalized abundance data from CSV
    with proper handling of multi-level headers (groups and sample IDs)
    """
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.raw_df = None
        self.metadata = None
        self.abundance_data = None
        self.groups = {}
        
    def load_data(self):
        """
        Load CSV file with proper multi-level header handling
        Manually parse the group structure to handle empty cells in group row
        """
        with open(self.filepath, 'r') as f:
            lines = [f.readline() for _ in range(3)]
        
        group_row = lines[1].strip().split(',')
        sample_row = lines[2].strip().split(',')
        
        print(f"\nDEBUG: Parsing CSV headers...")
        print(f"  Group row length: {len(group_row)}")
        print(f"  Sample row length: {len(sample_row)}")
        print(f"  First 20 groups: {group_row[:20]}")
        print(f"  Groups from col 14-20: {group_row[14:20]}")
        
        self.raw_df = pd.read_csv(self.filepath, skiprows=[0, 1, 2])
        
        metadata_cols = ['Compound', 'Neutral mass (Da)', 'm/z', 'Charge', 
                        'Retention time (min)', 'Chromatographic peak width (min)',
                        'Identifications', 'Anova (p)', 'Max Fold Change',
                        'Highest Mean', 'Lowest Mean', 'Isotope Distribution', 
                        'Maximum Abundance', 'Minimum CV%']
        
        self.metadata = self.raw_df.iloc[:, :14].copy()
        self.metadata.columns = metadata_cols
        
        self.abundance_data = self.raw_df.iloc[:, 14:].copy()
        
        new_columns = []
        for i, col_name in enumerate(self.abundance_data.columns):
            col_idx = i + 14
            if col_idx < len(group_row):
                group = group_row[col_idx].strip()
                sample = sample_row[col_idx].strip() if col_idx < len(sample_row) else col_name
                new_columns.append((group, sample))
            else:
                new_columns.append((col_name, col_name))
        
        self.abundance_data.columns = pd.MultiIndex.from_tuples(new_columns)
        
        self._parse_groups()
        
        return self.metadata, self.abundance_data, self.groups
    
    def _parse_groups(self):
        """
        Parse multi-level column headers to extract group assignments
        Now that we have proper (group, sample_id) tuples
        """
        print(f"\nDEBUG: First 5 abundance columns after manual parsing:")
        for i, col in enumerate(self.abundance_data.columns[:5]):
            print(f"  Column {i}: {col}")
        
        for col in self.abundance_data.columns:
            if isinstance(col, tuple) and len(col) >= 2:
                group = col[0].strip()
                
                if group and group != '' and group in ['LC', 'NC', 'PTB', 'QC']:
                    if group not in self.groups:
                        self.groups[group] = []
                    self.groups[group].append(col)
        
        print(f"\nDetected sample groups:")
        if self.groups:
            for group, samples in self.groups.items():
                print(f"  {group}: {len(samples)} samples")
        else:
            print("  ERROR: No groups detected!")


class MissingValueVisualizer:
    """
    Publication-ready visualization suite for missing value analysis
    in metabolomics data
    """
    
    def __init__(self, abundance_data, groups, metadata=None):
        self.abundance_data = abundance_data
        self.groups = groups
        self.metadata = metadata
        self.missing_mask = self._identify_missing_values(abundance_data)
        
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 11
        plt.rcParams['ytick.labelsize'] = 11
        plt.rcParams['legend.fontsize'] = 11
        plt.rcParams['figure.titlesize'] = 18
    
    def _identify_missing_values(self, data):
        """
        Identify missing values in multiple formats
        """
        missing = pd.DataFrame(False, index=data.index, columns=data.columns)
        
        for col in data.columns:
            col_data = data[col]
            missing[col] = (
                (col_data == 0) |
                (pd.isna(col_data)) |
                (col_data.astype(str).str.strip() == '') |
                (~pd.to_numeric(col_data, errors='coerce').notna())
            )
        
        return missing
        
    def plot_all(self, output_dir='missing_value_plots'):
        """
        Generate all missing value visualizations and save to directory
        """
        Path(output_dir).mkdir(exist_ok=True)
        
        fig_heatmap = self.plot_missing_heatmap()
        fig_heatmap.savefig(f'{output_dir}/01_missing_heatmap.png', 
                           dpi=300, bbox_inches='tight')
        
        fig_group = self.plot_missing_by_group()
        fig_group.savefig(f'{output_dir}/02_missing_by_group.png', 
                         dpi=300, bbox_inches='tight')
        
        fig_feature = self.plot_missing_by_feature()
        fig_feature.savefig(f'{output_dir}/03_missing_by_feature.png', 
                           dpi=300, bbox_inches='tight')
        
        fig_dist = self.plot_missing_distribution()
        fig_dist.savefig(f'{output_dir}/04_missing_distribution.png', 
                        dpi=300, bbox_inches='tight')
        
        fig_sample = self.plot_sample_quality()
        fig_sample.savefig(f'{output_dir}/05_sample_quality.png', 
                          dpi=300, bbox_inches='tight')
        
        stats = self.print_missing_statistics()
        with open(f'{output_dir}/missing_statistics.txt', 'w', encoding='utf-8') as f:
            f.write(stats)
        
        print(f"\n[OK] All plots saved to '{output_dir}/' directory")
        plt.close('all')
    
    def plot_missing_heatmap(self, max_features=100):
        """
        Heatmap showing missing value patterns across samples and features
        """
        missing_pct_per_feature = self.missing_mask.sum(axis=1) / len(self.missing_mask.columns) * 100
        sorted_indices = missing_pct_per_feature.sort_values(ascending=False).index[:max_features]
        
        plot_data = self.missing_mask.loc[sorted_indices]
        
        fig, ax = plt.subplots(figsize=(18, 10))
        
        sns.heatmap(plot_data.astype(int), 
                   cmap=['#27ae60', '#e74c3c'],
                   cbar_kws={'label': 'Missing (Red) vs Present (Green)', 'shrink': 0.8},
                   xticklabels=False,
                   yticklabels=False,
                   linewidths=0,
                   ax=ax)
        
        ax.set_xlabel('Samples →', fontsize=14, fontweight='bold')
        ax.set_ylabel(f'Features (Top {max_features} by missingness) →', 
                     fontsize=14, fontweight='bold')
        ax.set_title('Missing Value Pattern Across Samples and Features', 
                    fontsize=16, fontweight='bold', pad=20)
        
        group_positions = []
        current_pos = 0
        for group, cols in self.groups.items():
            mid_pos = current_pos + len(cols)/2
            group_positions.append((mid_pos, group))
            if current_pos > 0:
                ax.axvline(current_pos, color='white', linewidth=2, alpha=0.8)
            current_pos += len(cols)
        
        for pos, label in group_positions:
            ax.text(pos, -3, label, ha='center', va='top', 
                   fontsize=12, fontweight='bold', color='#2c3e50')
        
        plt.tight_layout()
        return fig
    
    def plot_missing_by_group(self):
        """
        Bar plot comparing missing value percentage across biological groups
        """
        group_missing = {}
        for group, cols in self.groups.items():
            group_mask = self.missing_mask[cols]
            missing_pct = group_mask.sum().sum() / group_mask.size * 100
            group_missing[group] = missing_pct
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        groups = list(group_missing.keys())
        values = list(group_missing.values())
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12'][:len(groups)]
        
        bars = ax.bar(groups, values, color=colors, edgecolor='black', 
                     linewidth=2, alpha=0.85, width=0.6)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{height:.2f}%',
                   ha='center', va='bottom', fontweight='bold', fontsize=13)
        
        ax.set_ylabel('Missing Values (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Group', fontsize=14, fontweight='bold')
        ax.set_title('Missing Value Percentage by Sample Group', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        ax.set_ylim(0, max(values) * 1.2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_missing_by_feature(self, bins=50):
        """
        Histogram showing distribution of missing values per feature
        """
        missing_per_feature = self.missing_mask.sum(axis=1)
        total_samples = len(self.abundance_data.columns)
        missing_pct_per_feature = (missing_per_feature / total_samples) * 100
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        n, bins_edges, patches = ax.hist(missing_pct_per_feature, bins=bins, 
                                         edgecolor='black', linewidth=1.5,
                                         color='#3498db', alpha=0.75)
        
        ax.set_xlabel('Missing Values per Feature (%)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Number of Features', fontsize=14, fontweight='bold')
        ax.set_title('Distribution of Missing Values Across Features', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        
        mean_missing = missing_pct_per_feature.mean()
        median_missing = missing_pct_per_feature.median()
        
        ax.axvline(mean_missing, color='red', linestyle='--', linewidth=2.5, 
                  label=f'Mean: {mean_missing:.2f}%', alpha=0.8)
        ax.axvline(median_missing, color='green', linestyle='--', linewidth=2.5, 
                  label=f'Median: {median_missing:.2f}%', alpha=0.8)
        ax.legend(fontsize=12, loc='upper right')
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_missing_distribution(self):
        """
        Shows feature distribution by missing value severity categories
        """
        missing_per_feature = self.missing_mask.sum(axis=1)
        total_samples = len(self.abundance_data.columns)
        missing_pct_per_feature = (missing_per_feature / total_samples) * 100
        
        categories = {
            'Complete\n(0%)': (missing_pct_per_feature == 0).sum(),
            'Low\n(<25%)': ((missing_pct_per_feature > 0) & (missing_pct_per_feature < 25)).sum(),
            'Medium\n(25-50%)': ((missing_pct_per_feature >= 25) & (missing_pct_per_feature < 50)).sum(),
            'High\n(50-75%)': ((missing_pct_per_feature >= 50) & (missing_pct_per_feature < 75)).sum(),
            'Very High\n(>=75%)': (missing_pct_per_feature >= 75).sum()
        }
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = ['#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
        bars = ax.bar(categories.keys(), categories.values(), 
                     color=colors, edgecolor='black', linewidth=2, alpha=0.85)
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{int(height)}',
                   ha='center', va='bottom', fontweight='bold', fontsize=13)
        
        ax.set_ylabel('Number of Features', fontsize=14, fontweight='bold')
        ax.set_xlabel('Missing Value Category', fontsize=14, fontweight='bold')
        ax.set_title('Feature Distribution by Missing Value Severity', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def plot_sample_quality(self):
        """
        Bar plot showing missing percentage for each sample by group
        """
        sample_missing = {}
        sample_groups = {}
        
        for group, cols in self.groups.items():
            for col in cols:
                sample_id = col[1] if isinstance(col, tuple) else col
                missing_pct = self.missing_mask[col].sum() / len(self.missing_mask) * 100
                sample_missing[sample_id] = missing_pct
                sample_groups[sample_id] = group
        
        sorted_samples = sorted(sample_missing.items(), key=lambda x: x[1], reverse=True)
        samples, values = zip(*sorted_samples)
        
        group_colors = {'LC': '#3498db', 'NC': '#2ecc71', 
                       'PTB': '#e74c3c', 'QC': '#f39c12'}
        colors = [group_colors.get(sample_groups[s], '#95a5a6') for s in samples]
        
        fig, ax = plt.subplots(figsize=(18, 7))
        
        bars = ax.bar(range(len(samples)), values, color=colors, 
                     edgecolor='black', linewidth=0.8, alpha=0.85)
        
        ax.set_ylabel('Missing Values (%)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index (Sorted by Missing %)', fontsize=14, fontweight='bold')
        ax.set_title('Sample Quality: Missing Value Percentage per Sample', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=1)
        
        threshold = 50
        ax.axhline(threshold, color='red', linestyle='--', linewidth=2, 
                  label=f'Quality Threshold ({threshold}%)', alpha=0.7)
        
        bad_samples = [s for s, v in zip(samples, values) if v > threshold]
        if bad_samples:
            for i, (s, v) in enumerate(zip(samples, values)):
                if v > threshold:
                    ax.text(i, v + 2, s, rotation=90, ha='center', va='bottom',
                           fontsize=9, color='red', fontweight='bold')
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, edgecolor='black', label=group)
                          for group, color in group_colors.items()]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return fig
    
    def print_missing_statistics(self):
        """
        Comprehensive statistical summary with quality flags
        """
        total_values = self.abundance_data.size
        total_missing = self.missing_mask.sum().sum()
        overall_missing_pct = (total_missing / total_values) * 100
        
        stats = []
        stats.append("=" * 70)
        stats.append("MISSING VALUE ANALYSIS REPORT - PUBLICATION QUALITY")
        stats.append("=" * 70)
        stats.append(f"\nDataset Overview:")
        stats.append(f"  Total features: {len(self.abundance_data)}")
        stats.append(f"  Total samples: {len(self.abundance_data.columns)}")
        stats.append(f"  Total data points: {total_values:,}")
        stats.append(f"\nOverall Missing Values:")
        stats.append(f"  Count: {total_missing:,}")
        stats.append(f"  Percentage: {overall_missing_pct:.2f}%")
        
        stats.append(f"\nMissing Values by Biological Group:")
        for group, cols in self.groups.items():
            group_mask = self.missing_mask[cols]
            group_missing = group_mask.sum().sum()
            group_total = group_mask.size
            group_pct = (group_missing / group_total) * 100
            stats.append(f"  {group}:")
            stats.append(f"    Samples: {len(cols)}")
            stats.append(f"    Missing: {group_missing:,} ({group_pct:.2f}%)")
        
        missing_per_feature = self.missing_mask.sum(axis=1)
        total_samples = len(self.abundance_data.columns)
        missing_pct_per_feature = (missing_per_feature / total_samples) * 100
        
        stats.append(f"\nFeature-level Statistics:")
        stats.append(f"  Complete features (0% missing): {(missing_pct_per_feature == 0).sum()}")
        stats.append(f"  Low missing (<25%): {((missing_pct_per_feature > 0) & (missing_pct_per_feature < 25)).sum()}")
        stats.append(f"  Medium missing (25-50%): {((missing_pct_per_feature >= 25) & (missing_pct_per_feature < 50)).sum()}")
        stats.append(f"  High missing (50-75%): {((missing_pct_per_feature >= 50) & (missing_pct_per_feature < 75)).sum()}")
        stats.append(f"  Very high missing (>=75%): {(missing_pct_per_feature >= 75).sum()}")
        stats.append(f"  Mean missing per feature: {missing_pct_per_feature.mean():.2f}%")
        stats.append(f"  Median missing per feature: {missing_pct_per_feature.median():.2f}%")
        
        sample_missing_pct = {}
        for group, cols in self.groups.items():
            for col in cols:
                sample_id = col[1] if isinstance(col, tuple) else col
                missing_pct = self.missing_mask[col].sum() / len(self.missing_mask) * 100
                sample_missing_pct[sample_id] = (missing_pct, group)
        
        bad_samples = [(s, v, g) for s, (v, g) in sample_missing_pct.items() if v > 50]
        if bad_samples:
            stats.append(f"\n! QUALITY FLAGS - Samples with >50% Missing:")
            for sample, pct, group in sorted(bad_samples, key=lambda x: x[1], reverse=True):
                stats.append(f"  * {sample} ({group}): {pct:.2f}% missing")
                stats.append(f"    -> RECOMMENDATION: Consider excluding from analysis")
        
        stats.append("\n" + "=" * 70)
        
        report = "\n".join(stats)
        print(report)
        return report


def main():
    """
    Main execution workflow for missing value visualization
    """
    csv_file = '../../data/progenesis_file.csv'
    
    print("=" * 70)
    print("MISSING VALUE ANALYSIS - PUBLICATION READY")
    print("=" * 70)
    print(f"\nLoading data from: {csv_file}")
    
    loader = ProgenesisDataLoader(csv_file)
    metadata, abundance_data, groups = loader.load_data()
    
    print("\nGenerating publication-quality visualizations...")
    visualizer = MissingValueVisualizer(abundance_data, groups, metadata)
    visualizer.plot_all(output_dir='missing_value_analysis')
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()