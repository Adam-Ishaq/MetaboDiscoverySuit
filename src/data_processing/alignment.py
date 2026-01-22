"""
MetaboAI - Feature Alignment Module
Aligns features across multiple samples to create a unified feature table.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import logging
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureAligner:
    """
    Aligns features across multiple LC-MS samples.
    
    This class handles:
    - Loading individual feature files
    - Grouping features within tolerance windows
    - Creating aligned feature table
    - Handling missing values
    
    Attributes:
        mz_tolerance (float): m/z tolerance in Daltons
        rt_tolerance (float): Retention time tolerance in seconds
        min_fraction (float): Minimum fraction of samples feature must appear in
    """
    
    def __init__(self, 
                 mz_tolerance: float = 0.01,
                 rt_tolerance: float = 30.0,
                 min_fraction: float = 0.3):
        """
        Initialize the feature aligner.
        
        Args:
            mz_tolerance (float): m/z tolerance for matching (default: 0.01 Da)
            rt_tolerance (float): RT tolerance for matching (default: 30 seconds)
            min_fraction (float): Minimum fraction of samples (default: 0.3 = 30%)
        """
        self.mz_tolerance = mz_tolerance
        self.rt_tolerance = rt_tolerance
        self.min_fraction = min_fraction
        
        self.feature_files = []
        self.sample_ids = []
        self.aligned_features = None
        self.feature_table = None
        
        logger.info(f"FeatureAligner initialized:")
        logger.info(f"  m/z tolerance: {mz_tolerance} Da")
        logger.info(f"  RT tolerance: {rt_tolerance} seconds")
        logger.info(f"  Min fraction: {min_fraction}")
    
    def load_feature_files(self, 
                          feature_dir: str,
                          metadata_path: Optional[str] = None) -> int:
        """
        Load all feature files from a directory.
        
        Args:
            feature_dir (str): Directory containing feature CSV files
            metadata_path (str, optional): Path to metadata file for sample order
            
        Returns:
            int: Number of feature files loaded
        """
        feature_dir = Path(feature_dir)
        
        if not feature_dir.exists():
            logger.error(f"Feature directory not found: {feature_dir}")
            return 0
        
        # Get all feature CSV files
        feature_files = sorted(feature_dir.glob("*_features.csv"))
        
        if len(feature_files) == 0:
            logger.error(f"No feature files found in {feature_dir}")
            return 0
        
        # If metadata provided, order files accordingly
        if metadata_path:
            metadata = pd.read_csv(metadata_path)
            sample_ids = metadata['sample_id'].tolist()
            
            # Reorder feature files to match metadata
            ordered_files = []
            for sample_id in sample_ids:
                matching_file = feature_dir / f"{sample_id}_features.csv"
                if matching_file.exists():
                    ordered_files.append(matching_file)
            
            feature_files = ordered_files
        
        self.feature_files = feature_files
        self.sample_ids = [f.stem.replace('_features', '') for f in feature_files]
        
        logger.info(f"Loaded {len(feature_files)} feature files")
        logger.info(f"Samples: {', '.join(self.sample_ids[:5])}...")
        
        return len(feature_files)
    
    def _cluster_features_grid_based(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Cluster features using fast grid-based binning.
        Much faster than hierarchical clustering for large datasets.
        
        Args:
            features_df (DataFrame): DataFrame with m/z and rt columns
            
        Returns:
            DataFrame: Features with cluster assignments
        """
        if len(features_df) == 0:
            return features_df
        
        # Create bins for m/z and RT
        mz_bins = np.arange(
            features_df['mz'].min() - self.mz_tolerance,
            features_df['mz'].max() + self.mz_tolerance,
            self.mz_tolerance
        )
        
        rt_bins = np.arange(
            features_df['rt'].min() - self.rt_tolerance,
            features_df['rt'].max() + self.rt_tolerance,
            self.rt_tolerance
        )
        
        # Assign features to bins
        mz_bin_idx = np.digitize(features_df['mz'].values, mz_bins)
        rt_bin_idx = np.digitize(features_df['rt'].values, rt_bins)
        
        # Create cluster ID from bin combination
        features_df['cluster_id'] = mz_bin_idx * 100000 + rt_bin_idx
        
        logger.info(f"  Grid-based clustering: {features_df['cluster_id'].nunique()} bins created")
        
        return features_df
    
    def align_features(self, max_features_per_sample: Optional[int] = None) -> pd.DataFrame:
        """
        Align features across all samples.
        
        Args:
            max_features_per_sample (int, optional): Limit features per sample for memory
            
        Returns:
            DataFrame: Aligned feature list with consensus m/z and RT
        """
        if len(self.feature_files) == 0:
            logger.error("No feature files loaded. Call load_feature_files() first.")
            return pd.DataFrame()
        
        logger.info("=" * 70)
        logger.info("Starting Feature Alignment")
        logger.info("=" * 70)
        
        # Step 1: Load and combine all features
        logger.info("Step 1: Loading all features...")
        all_features = []
        
        for sample_id, feature_file in zip(tqdm(self.sample_ids, desc="Loading samples"), 
                                           self.feature_files):
            try:
                features = pd.read_csv(feature_file)
                
                # Limit features if specified
                if max_features_per_sample and len(features) > max_features_per_sample:
                    features = features.nlargest(max_features_per_sample, 'intensity')
                
                features['sample_id'] = sample_id
                all_features.append(features)
                
            except Exception as e:
                logger.error(f"Error loading {feature_file}: {e}")
        
        if len(all_features) == 0:
            logger.error("No features loaded successfully")
            return pd.DataFrame()
        
        combined_features = pd.concat(all_features, ignore_index=True)
        logger.info(f"  Loaded {len(combined_features):,} total features")
        
        # Step 2: Cluster features
        logger.info("Step 2: Clustering features...")
        combined_features = self._cluster_features_grid_based(combined_features)
        
        n_clusters = combined_features['cluster_id'].nunique()
        logger.info(f"  Found {n_clusters:,} feature clusters")
        
        # Step 3: Create consensus features
        logger.info("Step 3: Creating consensus features...")
        
        aligned_features = []
        
        for cluster_id in tqdm(combined_features['cluster_id'].unique(), 
                              desc="Processing clusters"):
            cluster_data = combined_features[combined_features['cluster_id'] == cluster_id]
            
            # Calculate consensus m/z and RT (median)
            consensus_mz = cluster_data['mz'].median()
            consensus_rt = cluster_data['rt'].median()
            
            # Count how many samples this feature appears in
            n_samples = cluster_data['sample_id'].nunique()
            fraction = n_samples / len(self.sample_ids)
            
            # Filter by minimum fraction
            if fraction >= self.min_fraction:
                # Get intensity for each sample
                sample_intensities = {}
                for sample_id in self.sample_ids:
                    sample_data = cluster_data[cluster_data['sample_id'] == sample_id]
                    if len(sample_data) > 0:
                        # Use maximum intensity if multiple peaks in cluster
                        sample_intensities[sample_id] = sample_data['intensity'].max()
                    else:
                        sample_intensities[sample_id] = 0.0
                
                feature_info = {
                    'feature_id': f"F_{consensus_mz:.4f}_{consensus_rt:.1f}",
                    'mz': consensus_mz,
                    'rt': consensus_rt,
                    'n_samples': n_samples,
                    'fraction': fraction,
                    **sample_intensities
                }
                
                aligned_features.append(feature_info)
        
        self.aligned_features = pd.DataFrame(aligned_features)
        
        if len(self.aligned_features) == 0:
            logger.warning("No features passed the min_fraction filter!")
            logger.warning(f"Try lowering min_fraction (current: {self.min_fraction})")
            return self.aligned_features
        
        logger.info(f"  Created {len(self.aligned_features):,} aligned features")
        
        # Calculate features per sample (safely)
        intensity_cols = [col for col in self.aligned_features.columns if col in self.sample_ids]
        if intensity_cols:
            avg_features = self.aligned_features[intensity_cols].astype(bool).sum().mean()
            logger.info(f"  Features per sample: {avg_features:.0f}")
        
        return self.aligned_features
    
    def create_feature_table(self, 
                           fill_missing: str = 'zero',
                           log_transform: bool = False) -> pd.DataFrame:
        """
        Create the final feature table (samples × features).
        
        Args:
            fill_missing (str): How to handle missing values ('zero', 'min', 'median')
            log_transform (bool): Apply log2 transformation
            
        Returns:
            DataFrame: Feature table with samples as rows, features as columns
        """
        if self.aligned_features is None:
            logger.error("No aligned features. Call align_features() first.")
            return pd.DataFrame()
        
        logger.info("Creating feature table...")
        
        # Extract intensity columns (sample_ids)
        intensity_cols = self.sample_ids
        feature_table = self.aligned_features[['feature_id'] + intensity_cols].copy()
        
        # Transpose: samples as rows, features as columns
        feature_table = feature_table.set_index('feature_id').T
        feature_table.index.name = 'sample_id'
        
        # Handle missing values
        if fill_missing == 'zero':
            pass  # Already zeros
        elif fill_missing == 'min':
            min_val = feature_table[feature_table > 0].min().min()
            feature_table = feature_table.replace(0, min_val / 2)
        elif fill_missing == 'median':
            for col in feature_table.columns:
                col_median = feature_table[col][feature_table[col] > 0].median()
                feature_table[col] = feature_table[col].replace(0, col_median)
        
        # Log transformation
        if log_transform:
            feature_table = np.log2(feature_table + 1)
        
        self.feature_table = feature_table
        
        logger.info(f"Feature table created: {feature_table.shape[0]} samples × {feature_table.shape[1]} features")
        
        return feature_table
    
    def get_feature_metadata(self) -> pd.DataFrame:
        """
        Get metadata about aligned features.
        
        Returns:
            DataFrame: Feature metadata (m/z, RT, detection frequency)
        """
        if self.aligned_features is None:
            return pd.DataFrame()
        
        metadata = self.aligned_features[['feature_id', 'mz', 'rt', 'n_samples', 'fraction']].copy()
        
        # Add statistics
        intensity_cols = [col for col in self.aligned_features.columns if col in self.sample_ids]
        metadata['mean_intensity'] = self.aligned_features[intensity_cols].mean(axis=1)
        metadata['median_intensity'] = self.aligned_features[intensity_cols].median(axis=1)
        metadata['cv'] = (self.aligned_features[intensity_cols].std(axis=1) / 
                         self.aligned_features[intensity_cols].mean(axis=1) * 100)
        
        return metadata
    
    def save_results(self, output_dir: str):
        """
        Save alignment results to files.
        
        Args:
            output_dir (str): Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.feature_table is not None:
            # Save feature table
            table_path = output_dir / "feature_table.csv"
            self.feature_table.to_csv(table_path)
            logger.info(f"✓ Feature table saved: {table_path}")
            
            # Save transposed version (features as rows)
            transposed_path = output_dir / "feature_table_transposed.csv"
            self.feature_table.T.to_csv(transposed_path)
            logger.info(f"✓ Transposed table saved: {transposed_path}")
        
        if self.aligned_features is not None:
            # Save feature metadata
            metadata = self.get_feature_metadata()
            metadata_path = output_dir / "feature_metadata.csv"
            metadata.to_csv(metadata_path, index=False)
            logger.info(f"✓ Feature metadata saved: {metadata_path}")
        
        logger.info(f"All results saved to: {output_dir}")
    
    def get_alignment_stats(self) -> Dict:
        """
        Get alignment statistics.
        
        Returns:
            dict: Dictionary with alignment statistics
        """
        if self.feature_table is None:
            return {}
        
        stats = {
            'n_samples': len(self.sample_ids),
            'n_features': self.feature_table.shape[1],
            'n_total_features_before': sum([len(pd.read_csv(f)) for f in self.feature_files]),
            'compression_ratio': sum([len(pd.read_csv(f)) for f in self.feature_files]) / self.feature_table.shape[1],
            'missing_values': (self.feature_table == 0).sum().sum(),
            'missing_percentage': (self.feature_table == 0).sum().sum() / self.feature_table.size * 100,
            'mean_features_per_sample': (self.feature_table > 0).sum(axis=1).mean(),
        }
        
        return stats


def align_malaria_dataset(feature_dir: str = "results/processed/individual_features",
                         metadata_path: str = "data/metadata/malaria_metadata.csv",
                         output_dir: str = "results/aligned",
                         mz_tolerance: float = 0.01,
                         rt_tolerance: float = 30.0,
                         min_fraction: float = 0.3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to align the malaria dataset.
    
    Args:
        feature_dir (str): Directory with individual feature files
        metadata_path (str): Path to metadata CSV
        output_dir (str): Output directory
        mz_tolerance (float): m/z tolerance
        rt_tolerance (float): RT tolerance  
        min_fraction (float): Minimum fraction of samples
        
    Returns:
        tuple: (feature_table, feature_metadata)
    """
    # Initialize aligner
    aligner = FeatureAligner(
        mz_tolerance=mz_tolerance,
        rt_tolerance=rt_tolerance,
        min_fraction=min_fraction
    )
    
    # Load feature files
    n_files = aligner.load_feature_files(feature_dir, metadata_path)
    
    if n_files == 0:
        return pd.DataFrame(), pd.DataFrame()
    
    # Align features
    aligned_features = aligner.align_features()
    
    # Create feature table
    feature_table = aligner.create_feature_table(fill_missing='zero')
    
    # Get metadata
    feature_metadata = aligner.get_feature_metadata()
    
    # Save results
    aligner.save_results(output_dir)
    
    # Print statistics
    stats = aligner.get_alignment_stats()
    
    print("\n" + "=" * 70)
    print("Alignment Statistics")
    print("=" * 70)
    print(f"Samples:                    {stats['n_samples']}")
    print(f"Features before alignment:  {stats['n_total_features_before']:,}")
    print(f"Features after alignment:   {stats['n_features']:,}")
    print(f"Compression ratio:          {stats['compression_ratio']:.1f}x")
    print(f"Missing values:             {stats['missing_values']:,} ({stats['missing_percentage']:.1f}%)")
    print(f"Avg features per sample:    {stats['mean_features_per_sample']:.0f}")
    print("=" * 70)
    
    return feature_table, feature_metadata


# Example usage
if __name__ == "__main__":
    print("Feature Alignment Module")
    print("=" * 70)
    print()
    print("This module aligns features across LC-MS samples.")
    print()
    print("Usage:")
    print("  from src.data_processing.alignment import align_malaria_dataset")
    print()
    print("  feature_table, metadata = align_malaria_dataset()")
    print()
    print("=" * 70)