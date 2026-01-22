"""
MetaboAI - Statistical Analysis Module
Performs statistical analysis on preprocessed metabolomics data.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import logging
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """
    Performs statistical analysis on metabolomics data.
    
    Capabilities:
    - PCA for sample clustering
    - t-tests for differential analysis
    - Fold change calculation
    - Multiple testing correction
    - Results ranking and filtering
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.data = None
        self.metadata = None
        self.feature_metadata = None
        self.results = {}
        
        logger.info("StatisticalAnalyzer initialized")
    
    def load_data(self,
                  feature_table_path: str,
                  sample_metadata_path: str,
                  feature_metadata_path: Optional[str] = None):
        """
        Load preprocessed data and metadata.
        
        Args:
            feature_table_path (str): Path to preprocessed feature table
            sample_metadata_path (str): Path to sample metadata
            feature_metadata_path (str, optional): Path to feature metadata
        """
        logger.info("Loading data...")
        
        # Load feature table
        self.data = pd.read_csv(feature_table_path, index_col=0)
        logger.info(f"  Feature table: {self.data.shape[0]} samples × {self.data.shape[1]} features")
        
        # Load sample metadata
        self.metadata = pd.read_csv(sample_metadata_path)
        if 'sample_id' in self.metadata.columns:
            self.metadata = self.metadata.set_index('sample_id')
        logger.info(f"  Sample metadata: {len(self.metadata)} samples")
        
        # Load feature metadata
        if feature_metadata_path and Path(feature_metadata_path).exists():
            self.feature_metadata = pd.read_csv(feature_metadata_path)
            logger.info(f"  Feature metadata: {len(self.feature_metadata)} features")
        
        # Ensure metadata matches data
        common_samples = self.data.index.intersection(self.metadata.index)
        self.data = self.data.loc[common_samples]
        self.metadata = self.metadata.loc[common_samples]
        
        logger.info(f"✓ Loaded {len(common_samples)} matched samples")
    
    def perform_pca(self, n_components: int = 5) -> Tuple[pd.DataFrame, PCA]:
        """
        Perform Principal Component Analysis.
        
        Args:
            n_components (int): Number of principal components
            
        Returns:
            tuple: (PC scores DataFrame, PCA object)
        """
        logger.info(f"Performing PCA (n_components={n_components})...")
        
        # Perform PCA
        pca = PCA(n_components=n_components)
        pc_scores = pca.fit_transform(self.data)
        
        # Create DataFrame
        pc_df = pd.DataFrame(
            pc_scores,
            index=self.data.index,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Add metadata
        pc_df = pd.concat([pc_df, self.metadata], axis=1)
        
        # Calculate variance explained
        var_explained = pca.explained_variance_ratio_ * 100
        
        logger.info("PCA Results:")
        for i, var in enumerate(var_explained):
            logger.info(f"  PC{i+1}: {var:.2f}% variance explained")
        
        self.results['pca'] = {
            'scores': pc_df,
            'pca_object': pca,
            'variance_explained': var_explained
        }
        
        return pc_df, pca
    
    def differential_analysis(self,
                            group_column: str = 'group',
                            group1: str = 'Naive',
                            group2: str = 'Semi',
                            test_type: str = 'ttest') -> pd.DataFrame:
        """
        Perform differential analysis between two groups.
        
        Args:
            group_column (str): Column name for groups
            group1 (str): First group name
            group2 (str): Second group name
            test_type (str): 'ttest' or 'mannwhitney'
            
        Returns:
            DataFrame: Results with p-values, fold changes, etc.
        """
        logger.info(f"Performing differential analysis: {group1} vs {group2}")
        
        # Get samples for each group
        group1_samples = self.metadata[self.metadata[group_column] == group1].index
        group2_samples = self.metadata[self.metadata[group_column] == group2].index
        
        logger.info(f"  {group1}: {len(group1_samples)} samples")
        logger.info(f"  {group2}: {len(group2_samples)} samples")
        
        # Get data for each group
        group1_data = self.data.loc[group1_samples]
        group2_data = self.data.loc[group2_samples]
        
        results_list = []
        
        for feature in self.data.columns:
            vals1 = group1_data[feature].values
            vals2 = group2_data[feature].values
            
            # Calculate means
            mean1 = np.mean(vals1)
            mean2 = np.mean(vals2)
            
            # Calculate fold change
            # Add small constant to avoid division by zero
            fc = (mean2 + 1e-10) / (mean1 + 1e-10)
            log2fc = np.log2(fc)
            
            # Perform statistical test
            if test_type == 'ttest':
                stat, pval = stats.ttest_ind(vals1, vals2, equal_var=False)
            else:  # mannwhitney
                stat, pval = stats.mannwhitneyu(vals1, vals2, alternative='two-sided')
            
            results_list.append({
                'feature_id': feature,
                f'{group1}_mean': mean1,
                f'{group2}_mean': mean2,
                'fold_change': fc,
                'log2_fold_change': log2fc,
                'p_value': pval,
                'statistic': stat
            })
        
        results_df = pd.DataFrame(results_list)
        
        # Multiple testing correction (Benjamini-Hochberg FDR)
        from scipy.stats import false_discovery_control
        results_df['p_value_adjusted'] = false_discovery_control(results_df['p_value'].values)
        
        # Add significance flags
        results_df['significant_005'] = results_df['p_value_adjusted'] < 0.05
        results_df['significant_001'] = results_df['p_value_adjusted'] < 0.01
        
        # Sort by p-value
        results_df = results_df.sort_values('p_value')
        
        # Add feature metadata if available
        if self.feature_metadata is not None:
            results_df = results_df.merge(
                self.feature_metadata[['feature_id', 'mz', 'rt']],
                on='feature_id',
                how='left'
            )
        
        logger.info(f"✓ Differential analysis complete")
        logger.info(f"  Significant features (p<0.05): {results_df['significant_005'].sum()}")
        logger.info(f"  Significant features (p<0.01): {results_df['significant_001'].sum()}")
        
        self.results['differential'] = results_df
        
        return results_df
    
    def get_top_features(self,
                        n_features: int = 50,
                        criteria: str = 'pvalue') -> pd.DataFrame:
        """
        Get top differential features.
        
        Args:
            n_features (int): Number of top features
            criteria (str): 'pvalue', 'foldchange', or 'combined'
            
        Returns:
            DataFrame: Top features
        """
        if 'differential' not in self.results:
            raise ValueError("Run differential_analysis() first")
        
        df = self.results['differential'].copy()
        
        if criteria == 'pvalue':
            top_features = df.nsmallest(n_features, 'p_value_adjusted')
        elif criteria == 'foldchange':
            df['abs_log2fc'] = np.abs(df['log2_fold_change'])
            top_features = df.nlargest(n_features, 'abs_log2fc')
        elif criteria == 'combined':
            # Combined score: -log10(p) * abs(log2FC)
            df['combined_score'] = -np.log10(df['p_value_adjusted'] + 1e-300) * np.abs(df['log2_fold_change'])
            top_features = df.nlargest(n_features, 'combined_score')
        
        return top_features
    
    def export_results(self, output_dir: str):
        """
        Export all analysis results.
        
        Args:
            output_dir (str): Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export PCA scores
        if 'pca' in self.results:
            pca_path = output_dir / "pca_scores.csv"
            self.results['pca']['scores'].to_csv(pca_path)
            logger.info(f"✓ PCA scores saved: {pca_path}")
        
        # Export differential analysis results
        if 'differential' in self.results:
            diff_path = output_dir / "differential_analysis_results.csv"
            self.results['differential'].to_csv(diff_path, index=False)
            logger.info(f"✓ Differential analysis results saved: {diff_path}")
            
            # Export significant features only
            sig_features = self.results['differential'][
                self.results['differential']['significant_005']
            ]
            sig_path = output_dir / "significant_features.csv"
            sig_features.to_csv(sig_path, index=False)
            logger.info(f"✓ Significant features saved: {sig_path}")
            
            # Export top 50 features
            top_features = self.get_top_features(50, criteria='combined')
            top_path = output_dir / "top50_features.csv"
            top_features.to_csv(top_path, index=False)
            logger.info(f"✓ Top 50 features saved: {top_path}")
        
        logger.info(f"All results saved to: {output_dir}")
    
    def get_summary_statistics(self) -> dict:
        """Get summary statistics of the analysis."""
        summary = {}
        
        if 'differential' in self.results:
            df = self.results['differential']
            
            summary['total_features'] = len(df)
            summary['significant_005'] = df['significant_005'].sum()
            summary['significant_001'] = df['significant_001'].sum()
            summary['upregulated'] = ((df['log2_fold_change'] > 0) & 
                                     (df['significant_005'])).sum()
            summary['downregulated'] = ((df['log2_fold_change'] < 0) & 
                                       (df['significant_005'])).sum()
            summary['max_fold_change'] = df['fold_change'].max()
            summary['min_fold_change'] = df['fold_change'].min()
            summary['min_pvalue'] = df['p_value'].min()
        
        if 'pca' in self.results:
            summary['pca_variance_pc1'] = self.results['pca']['variance_explained'][0]
            summary['pca_variance_pc2'] = self.results['pca']['variance_explained'][1]
            summary['pca_total_variance'] = sum(self.results['pca']['variance_explained'])
        
        return summary


# Convenience function
def analyze_malaria_dataset(
    feature_table_path: str = "results/preprocessed/preprocessed_feature_table.csv",
    sample_metadata_path: str = "data/metadata/malaria_metadata.csv",
    feature_metadata_path: str = "results/preprocessed/preprocessed_feature_metadata.csv",
    output_dir: str = "results/statistical_analysis"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to analyze malaria dataset.
    
    Returns:
        tuple: (PCA scores, differential analysis results)
    """
    # Initialize analyzer
    analyzer = StatisticalAnalyzer()
    
    # Load data
    analyzer.load_data(
        feature_table_path=feature_table_path,
        sample_metadata_path=sample_metadata_path,
        feature_metadata_path=feature_metadata_path
    )
    
    # Perform PCA
    pca_scores, pca_obj = analyzer.perform_pca(n_components=5)
    
    # Perform differential analysis
    diff_results = analyzer.differential_analysis(
        group_column='group',
        group1='Naive',
        group2='Semi',
        test_type='ttest'
    )
    
    # Export results
    analyzer.export_results(output_dir)
    
    # Print summary
    summary = analyzer.get_summary_statistics()
    
    print("\n" + "=" * 70)
    print("STATISTICAL ANALYSIS SUMMARY")
    print("=" * 70)
    for key, value in summary.items():
        print(f"{key:30s}: {value}")
    print("=" * 70)
    
    return pca_scores, diff_results


# Example usage
if __name__ == "__main__":
    print("Statistical Analysis Module")
    print("=" * 70)
    print()
    print("This module performs statistical analysis on metabolomics data.")
    print()
    print("Usage:")
    print("  from src.statistical_analysis.statistical_analysis import analyze_malaria_dataset")
    print()
    print("  pca, diff = analyze_malaria_dataset()")
    print()
    print("=" * 70)