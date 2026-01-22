"""
MetaboAI - Preprocessing Module
Preprocesses aligned metabolomics data for statistical analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import logging
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MetabolomicsPreprocessor:
    """
    Preprocesses metabolomics data for statistical analysis.
    
    Steps:
    1. Quality filtering (missing values, low intensity)
    2. Normalization (total sum, median, etc.)
    3. Transformation (log, sqrt)
    4. Scaling (pareto, auto, range)
    5. Missing value imputation
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.original_data = None
        self.processed_data = None
        self.feature_metadata = None
        self.sample_metadata = None
        self.processing_log = []
        
        logger.info("MetabolomicsPreprocessor initialized")
    
    def load_data(self, 
                  feature_table_path: str,
                  feature_metadata_path: Optional[str] = None,
                  sample_metadata_path: Optional[str] = None):
        """
        Load feature table and metadata.
        
        Args:
            feature_table_path (str): Path to feature table CSV
            feature_metadata_path (str, optional): Path to feature metadata
            sample_metadata_path (str, optional): Path to sample metadata
        """
        logger.info("Loading data...")
        
        # Load feature table
        self.original_data = pd.read_csv(feature_table_path, index_col=0)
        self.processed_data = self.original_data.copy()
        
        logger.info(f"  Feature table: {self.original_data.shape[0]} samples × {self.original_data.shape[1]} features")
        
        # Load feature metadata
        if feature_metadata_path and Path(feature_metadata_path).exists():
            self.feature_metadata = pd.read_csv(feature_metadata_path)
            logger.info(f"  Feature metadata: {len(self.feature_metadata)} features")
        
        # Load sample metadata
        if sample_metadata_path and Path(sample_metadata_path).exists():
            self.sample_metadata = pd.read_csv(sample_metadata_path)
            # Set sample_id as index
            if 'sample_id' in self.sample_metadata.columns:
                self.sample_metadata = self.sample_metadata.set_index('sample_id')
            logger.info(f"  Sample metadata: {len(self.sample_metadata)} samples")
        
        self._log_step(f"Loaded data: {self.original_data.shape}")
    
    def _log_step(self, message: str):
        """Log a preprocessing step."""
        self.processing_log.append(message)
        logger.info(f"  ✓ {message}")
    
    def filter_missing_values(self, max_missing_percent: float = 50.0):
        """
        Remove features with too many missing values.
        
        Args:
            max_missing_percent (float): Maximum percentage of missing values allowed
        """
        logger.info(f"Filtering features by missing values (max: {max_missing_percent}%)...")
        
        n_samples = self.processed_data.shape[0]
        missing_percent = (self.processed_data == 0).sum() / n_samples * 100
        
        # Keep features with missing < threshold
        features_to_keep = missing_percent[missing_percent <= max_missing_percent].index
        
        n_removed = self.processed_data.shape[1] - len(features_to_keep)
        self.processed_data = self.processed_data[features_to_keep]
        
        # Update feature metadata
        if self.feature_metadata is not None:
            self.feature_metadata = self.feature_metadata[
                self.feature_metadata['feature_id'].isin(features_to_keep)
            ]
        
        self._log_step(f"Removed {n_removed} features (>{max_missing_percent}% missing). Remaining: {len(features_to_keep)}")
    
    def filter_low_intensity(self, min_intensity: float = 1000.0):
        """
        Remove features with low mean intensity.
        
        Args:
            min_intensity (float): Minimum mean intensity
        """
        logger.info(f"Filtering low intensity features (min: {min_intensity})...")
        
        # Calculate mean intensity (excluding zeros)
        mean_intensity = self.processed_data.apply(
            lambda x: x[x > 0].mean() if (x > 0).any() else 0
        )
        
        features_to_keep = mean_intensity[mean_intensity >= min_intensity].index
        
        n_removed = self.processed_data.shape[1] - len(features_to_keep)
        self.processed_data = self.processed_data[features_to_keep]
        
        # Update feature metadata
        if self.feature_metadata is not None:
            self.feature_metadata = self.feature_metadata[
                self.feature_metadata['feature_id'].isin(features_to_keep)
            ]
        
        self._log_step(f"Removed {n_removed} low intensity features. Remaining: {len(features_to_keep)}")
    
    def filter_by_detection_rate(self, min_detection_rate: float = 0.2):
        """
        Remove features detected in too few samples.
        
        Args:
            min_detection_rate (float): Minimum fraction of samples (0-1)
        """
        logger.info(f"Filtering by detection rate (min: {min_detection_rate*100}%)...")
        
        n_samples = self.processed_data.shape[0]
        detection_count = (self.processed_data > 0).sum()
        detection_rate = detection_count / n_samples
        
        features_to_keep = detection_rate[detection_rate >= min_detection_rate].index
        
        n_removed = self.processed_data.shape[1] - len(features_to_keep)
        self.processed_data = self.processed_data[features_to_keep]
        
        # Update feature metadata
        if self.feature_metadata is not None:
            self.feature_metadata = self.feature_metadata[
                self.feature_metadata['feature_id'].isin(features_to_keep)
            ]
        
        self._log_step(f"Removed {n_removed} rarely detected features. Remaining: {len(features_to_keep)}")
    
    def filter_by_rsd(self, max_rsd: float = 30.0, use_qc: bool = True):
        """
        Remove features with high RSD (Relative Standard Deviation) in QC samples.
        
        Args:
            max_rsd (float): Maximum RSD percentage
            use_qc (bool): Use QC samples if available
        """
        if not use_qc or self.sample_metadata is None:
            logger.warning("QC-based RSD filtering skipped (no sample metadata)")
            return
        
        logger.info(f"Filtering by RSD in QC samples (max: {max_rsd}%)...")
        
        # Find QC samples
        qc_samples = self.sample_metadata[self.sample_metadata['group'] == 'QC'].index
        
        if len(qc_samples) == 0:
            logger.warning("No QC samples found. Skipping RSD filter.")
            return
        
        # Calculate RSD in QC samples
        qc_data = self.processed_data.loc[qc_samples]
        qc_data_nonzero = qc_data.replace(0, np.nan)
        
        rsd = (qc_data_nonzero.std() / qc_data_nonzero.mean() * 100).fillna(100)
        
        features_to_keep = rsd[rsd <= max_rsd].index
        
        n_removed = self.processed_data.shape[1] - len(features_to_keep)
        self.processed_data = self.processed_data[features_to_keep]
        
        # Update feature metadata
        if self.feature_metadata is not None:
            self.feature_metadata = self.feature_metadata[
                self.feature_metadata['feature_id'].isin(features_to_keep)
            ]
        
        self._log_step(f"Removed {n_removed} unstable features (RSD>{max_rsd}%). Remaining: {len(features_to_keep)}")
    
    def normalize(self, method: str = 'median'):
        """
        Normalize samples.
        
        Args:
            method (str): Normalization method ('sum', 'median', 'quantile', 'none')
        """
        if method == 'none':
            logger.info("Skipping normalization")
            return
        
        logger.info(f"Normalizing data (method: {method})...")
        
        if method == 'sum':
            # Total sum normalization
            row_sums = self.processed_data.sum(axis=1)
            median_sum = row_sums.median()
            self.processed_data = self.processed_data.div(row_sums, axis=0) * median_sum
            
        elif method == 'median':
            # Median normalization
            row_medians = self.processed_data.replace(0, np.nan).median(axis=1)
            global_median = row_medians.median()
            normalization_factors = global_median / row_medians
            self.processed_data = self.processed_data.mul(normalization_factors, axis=0)
            
        elif method == 'quantile':
            # Quantile normalization (simple version)
            from scipy.stats import rankdata
            ranked = self.processed_data.rank(method='average', axis=0)
            sorted_means = np.sort(self.processed_data.values, axis=0).mean(axis=1)
            
            for col in self.processed_data.columns:
                ranks = rankdata(self.processed_data[col], method='average')
                self.processed_data[col] = sorted_means[ranks.astype(int) - 1]
        
        self._log_step(f"Normalized using {method} method")
    
    def transform(self, method: str = 'log2'):
        """
        Transform data.
        
        Args:
            method (str): Transformation method ('log2', 'log10', 'sqrt', 'none')
        """
        if method == 'none':
            logger.info("Skipping transformation")
            return
        
        logger.info(f"Transforming data (method: {method})...")
        
        if method == 'log2':
            self.processed_data = np.log2(self.processed_data + 1)
        elif method == 'log10':
            self.processed_data = np.log10(self.processed_data + 1)
        elif method == 'sqrt':
            self.processed_data = np.sqrt(self.processed_data)
        
        self._log_step(f"Transformed using {method}")
    
    def scale(self, method: str = 'pareto'):
        """
        Scale features.
        
        Args:
            method (str): Scaling method ('auto', 'pareto', 'range', 'none')
        """
        if method == 'none':
            logger.info("Skipping scaling")
            return
        
        logger.info(f"Scaling features (method: {method})...")
        
        if method == 'auto':
            # Auto scaling (mean-centering + unit variance)
            scaler = StandardScaler()
            self.processed_data = pd.DataFrame(
                scaler.fit_transform(self.processed_data),
                index=self.processed_data.index,
                columns=self.processed_data.columns
            )
            
        elif method == 'pareto':
            # Pareto scaling (mean-centering + square root of std)
            means = self.processed_data.mean()
            stds = self.processed_data.std()
            self.processed_data = (self.processed_data - means) / np.sqrt(stds)
            
        elif method == 'range':
            # Range scaling
            mins = self.processed_data.min()
            maxs = self.processed_data.max()
            self.processed_data = (self.processed_data - mins) / (maxs - mins)
        
        self._log_step(f"Scaled using {method} method")
    
    def impute_missing_values(self, method: str = 'knn', n_neighbors: int = 5):
        """
        Impute missing values.
        
        Args:
            method (str): Imputation method ('knn', 'min', 'median', 'zero')
            n_neighbors (int): Number of neighbors for KNN
        """
        logger.info(f"Imputing missing values (method: {method})...")
        
        # Replace 0 with NaN for imputation
        data_with_nan = self.processed_data.replace(0, np.nan)
        
        if method == 'knn':
            imputer = KNNImputer(n_neighbors=n_neighbors)
            imputed = imputer.fit_transform(data_with_nan)
            self.processed_data = pd.DataFrame(
                imputed,
                index=self.processed_data.index,
                columns=self.processed_data.columns
            )
            
        elif method == 'min':
            # Replace with half of minimum non-zero value
            min_val = self.processed_data[self.processed_data > 0].min().min()
            self.processed_data = data_with_nan.fillna(min_val / 2)
            
        elif method == 'median':
            # Replace with column median
            self.processed_data = data_with_nan.fillna(data_with_nan.median())
            
        elif method == 'zero':
            # Keep as zero
            pass
        
        self._log_step(f"Imputed missing values using {method}")
    
    def get_processing_summary(self) -> str:
        """Get a summary of all preprocessing steps."""
        summary = "\n" + "=" * 70 + "\n"
        summary += "PREPROCESSING SUMMARY\n"
        summary += "=" * 70 + "\n\n"
        
        summary += "Original data:\n"
        summary += f"  Samples: {self.original_data.shape[0]}\n"
        summary += f"  Features: {self.original_data.shape[1]}\n\n"
        
        summary += "Processed data:\n"
        summary += f"  Samples: {self.processed_data.shape[0]}\n"
        summary += f"  Features: {self.processed_data.shape[1]}\n"
        summary += f"  Features retained: {self.processed_data.shape[1]/self.original_data.shape[1]*100:.1f}%\n\n"
        
        summary += "Processing steps:\n"
        for i, step in enumerate(self.processing_log, 1):
            summary += f"  {i}. {step}\n"
        
        summary += "\n" + "=" * 70
        
        return summary
    
    def save_processed_data(self, output_dir: str):
        """
        Save preprocessed data.
        
        Args:
            output_dir (str): Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save processed feature table
        table_path = output_dir / "preprocessed_feature_table.csv"
        self.processed_data.to_csv(table_path)
        logger.info(f"✓ Preprocessed feature table saved: {table_path}")
        
        # Save transposed version
        transposed_path = output_dir / "preprocessed_feature_table_transposed.csv"
        self.processed_data.T.to_csv(transposed_path)
        logger.info(f"✓ Transposed table saved: {transposed_path}")
        
        # Save updated feature metadata
        if self.feature_metadata is not None:
            metadata_path = output_dir / "preprocessed_feature_metadata.csv"
            self.feature_metadata.to_csv(metadata_path, index=False)
            logger.info(f"✓ Feature metadata saved: {metadata_path}")
        
        # Save processing log
        log_path = output_dir / "preprocessing_log.txt"
        with open(log_path, 'w') as f:
            f.write(self.get_processing_summary())
        logger.info(f"✓ Processing log saved: {log_path}")
        
        logger.info(f"All results saved to: {output_dir}")


# Example usage
if __name__ == "__main__":
    print("Preprocessing Module")
    print("=" * 70)
    print()
    print("This module preprocesses aligned metabolomics data.")
    print()
    print("Usage:")
    print("  from src.preprocessing.preprocessing import MetabolomicsPreprocessor")
    print()
    print("  preprocessor = MetabolomicsPreprocessor()")
    print("  preprocessor.load_data('feature_table.csv')")
    print("  preprocessor.filter_missing_values()")
    print("  preprocessor.normalize(method='median')")
    print("  preprocessor.transform(method='log2')")
    print("  preprocessor.save_processed_data('output/')")
    print()
    print("=" * 70)