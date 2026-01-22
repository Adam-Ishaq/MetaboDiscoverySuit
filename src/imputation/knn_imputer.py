import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from visualize_missing import ProgenesisDataLoader


class KNNImputationAnalyzer:
    """
    KNN-based imputation with validation and comparison metrics
    for metabolomics missing value analysis
    """
    
    def __init__(self, abundance_data, groups, metadata=None):
        self.abundance_data = abundance_data
        self.groups = groups
        self.metadata = metadata
        self.imputed_data = None
        self.validation_results = {}
        
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
    
    def _identify_missing_values(self, data):
        """Identify missing values (0, NaN, empty, non-numeric)"""
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
    
    def _prepare_data_for_imputation(self, data, replace_with_nan=True):
        """Convert data to numeric and replace missing values with NaN"""
        data_clean = data.copy()
        
        for col in data_clean.columns:
            data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
        
        if replace_with_nan:
            data_clean = data_clean.replace(0, np.nan)
        
        return data_clean
    
    def impute_groupwise(self, k=5, scale=True):
        """
        Perform KNN imputation separately for each biological group
        
        Args:
            k: Number of neighbors for KNN
            scale: Whether to standardize data before imputation
        """
        print(f"\n{'='*70}")
        print(f"KNN IMPUTATION - GROUP-WISE (k={k})")
        print(f"{'='*70}")
        
        self.imputed_data = self.abundance_data.copy()
        imputation_stats = {}
        
        for group, cols in self.groups.items():
            print(f"\nProcessing group: {group}")
            print(f"  Samples: {len(cols)}")
            
            group_data = self.abundance_data[cols].copy()
            group_data_clean = self._prepare_data_for_imputation(group_data)
            
            missing_before = group_data_clean.isna().sum().sum()
            total_values = group_data_clean.size
            missing_pct = (missing_before / total_values) * 100
            
            print(f"  Missing values: {missing_before} ({missing_pct:.2f}%)")
            
            if missing_before == 0:
                print(f"  No missing values to impute!")
                for col in cols:
                    self.imputed_data[col] = group_data_clean[col]
                continue
            
            features_with_data = group_data_clean.notna().any(axis=1)
            group_data_filtered = group_data_clean.loc[features_with_data]
            
            if len(group_data_filtered) == 0:
                print(f"  No features with data in this group!")
                continue
            
            if scale:
                scaler = StandardScaler()
                group_data_transposed = group_data_filtered.T
                group_data_scaled = scaler.fit_transform(group_data_transposed)
            else:
                group_data_scaled = group_data_filtered.T.values
            
            imputer = KNNImputer(n_neighbors=k, weights='distance')
            imputed_values = imputer.fit_transform(group_data_scaled)
            
            if scale:
                imputed_values = scaler.inverse_transform(imputed_values)
            
            imputed_df = pd.DataFrame(
                imputed_values.T,
                index=group_data_filtered.index,
                columns=group_data_filtered.columns
            )
            
            result_df = group_data_clean.copy()
            result_df.loc[features_with_data] = imputed_df
            
            for col in cols:
                self.imputed_data[col] = result_df[col]
            
            missing_after = np.isnan(imputed_values).sum()
            imputed_count = missing_before - missing_after
            
            print(f"  Imputed: {imputed_count} values")
            
            imputation_stats[group] = {
                'samples': len(cols),
                'missing_before': missing_before,
                'imputed': imputed_count,
                'missing_pct': missing_pct
            }
        
        print(f"\n{'='*70}")
        print("IMPUTATION COMPLETE")
        print(f"{'='*70}\n")
        
        return imputation_stats
    
    def validate_imputation(self, mask_fraction=0.1, k_values=[3, 5, 7, 10, 15], n_trials=5):
        """
        Validate imputation by artificially masking known values
        
        Args:
            mask_fraction: Fraction of non-missing values to mask
            k_values: List of k values to test
            n_trials: Number of validation trials
        """
        print(f"\n{'='*70}")
        print(f"VALIDATION: Testing k values {k_values}")
        print(f"{'='*70}")
        
        validation_results = {k: {'rmse': [], 'mae': [], 'r2': [], 'correlation': []} 
                             for k in k_values}
        
        for trial in range(n_trials):
            print(f"\nTrial {trial + 1}/{n_trials}")
            
            data_clean = self._prepare_data_for_imputation(self.abundance_data)
            
            non_missing_mask = ~data_clean.isna()
            non_missing_indices = np.where(non_missing_mask.values)
            
            n_to_mask = int(len(non_missing_indices[0]) * mask_fraction)
            mask_indices = np.random.choice(len(non_missing_indices[0]), 
                                           size=n_to_mask, replace=False)
            
            rows_to_mask = non_missing_indices[0][mask_indices]
            cols_to_mask = non_missing_indices[1][mask_indices]
            
            true_values = []
            for r, c in zip(rows_to_mask, cols_to_mask):
                true_values.append(data_clean.iloc[r, c])
            true_values = np.array(true_values)
            
            masked_data = data_clean.copy()
            for r, c in zip(rows_to_mask, cols_to_mask):
                masked_data.iloc[r, c] = np.nan
            
            for k in k_values:
                scaler = StandardScaler()
                masked_transposed = masked_data.T
                masked_scaled = scaler.fit_transform(masked_transposed)
                
                imputer = KNNImputer(n_neighbors=k, weights='distance')
                imputed_scaled = imputer.fit_transform(masked_scaled)
                imputed_transposed = scaler.inverse_transform(imputed_scaled)
                imputed = imputed_transposed.T
                
                predicted_values = []
                for r, c in zip(rows_to_mask, cols_to_mask):
                    predicted_values.append(imputed[r, c])
                predicted_values = np.array(predicted_values)
                
                rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
                mae = mean_absolute_error(true_values, predicted_values)
                r2 = r2_score(true_values, predicted_values)
                corr, _ = pearsonr(true_values, predicted_values)
                
                validation_results[k]['rmse'].append(rmse)
                validation_results[k]['mae'].append(mae)
                validation_results[k]['r2'].append(r2)
                validation_results[k]['correlation'].append(corr)
        
        self.validation_results = {}
        for k in k_values:
            self.validation_results[k] = {
                'rmse_mean': np.mean(validation_results[k]['rmse']),
                'rmse_std': np.std(validation_results[k]['rmse']),
                'mae_mean': np.mean(validation_results[k]['mae']),
                'mae_std': np.std(validation_results[k]['mae']),
                'r2_mean': np.mean(validation_results[k]['r2']),
                'r2_std': np.std(validation_results[k]['r2']),
                'correlation_mean': np.mean(validation_results[k]['correlation']),
                'correlation_std': np.std(validation_results[k]['correlation'])
            }
        
        print(f"\n{'='*70}")
        print("VALIDATION RESULTS")
        print(f"{'='*70}")
        for k in k_values:
            print(f"\nk = {k}:")
            print(f"  RMSE: {self.validation_results[k]['rmse_mean']:.4f} +/- {self.validation_results[k]['rmse_std']:.4f}")
            print(f"  MAE:  {self.validation_results[k]['mae_mean']:.4f} +/- {self.validation_results[k]['mae_std']:.4f}")
            print(f"  R2:   {self.validation_results[k]['r2_mean']:.4f} +/- {self.validation_results[k]['r2_std']:.4f}")
            print(f"  Corr: {self.validation_results[k]['correlation_mean']:.4f} +/- {self.validation_results[k]['correlation_std']:.4f}")
        
        best_k_rmse = min(k_values, key=lambda k: self.validation_results[k]['rmse_mean'])
        best_k_r2 = max(k_values, key=lambda k: self.validation_results[k]['r2_mean'])
        
        print(f"\nBest k by RMSE: {best_k_rmse}")
        print(f"Best k by R2: {best_k_r2}")
        print(f"{'='*70}\n")
        
        return best_k_rmse
    
    def plot_validation_results(self, output_dir='knn_imputation_results'):
        """Plot validation metrics for different k values"""
        Path(output_dir).mkdir(exist_ok=True)
        
        k_values = sorted(self.validation_results.keys())
        
        metrics = ['rmse', 'mae', 'r2', 'correlation']
        metric_names = ['RMSE', 'MAE', 'R² Score', 'Pearson Correlation']
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (metric, name) in enumerate(zip(metrics, metric_names)):
            means = [self.validation_results[k][f'{metric}_mean'] for k in k_values]
            stds = [self.validation_results[k][f'{metric}_std'] for k in k_values]
            
            axes[idx].errorbar(k_values, means, yerr=stds, marker='o', 
                              markersize=8, linewidth=2, capsize=5, capthick=2)
            axes[idx].set_xlabel('Number of Neighbors (k)', fontweight='bold')
            axes[idx].set_ylabel(name, fontweight='bold')
            axes[idx].set_title(f'{name} vs k', fontweight='bold', pad=10)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].set_xticks(k_values)
            
            if metric in ['rmse', 'mae']:
                best_k = k_values[np.argmin(means)]
            else:
                best_k = k_values[np.argmax(means)]
            axes[idx].axvline(best_k, color='red', linestyle='--', 
                             alpha=0.7, linewidth=2, label=f'Best k={best_k}')
            axes[idx].legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/validation_metrics.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_dir}/validation_metrics.png")
        plt.close()
    
    def plot_imputation_comparison(self, output_dir='knn_imputation_results'):
        """Compare distributions before and after imputation"""
        Path(output_dir).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        for idx, (group, cols) in enumerate(self.groups.items()):
            if idx >= 4:
                break
            
            ax = axes[idx // 2, idx % 2]
            
            original_data = self._prepare_data_for_imputation(
                self.abundance_data[cols], replace_with_nan=False)
            imputed_data = self.imputed_data[cols]
            
            original_flat = original_data.values.flatten()
            original_flat = original_flat[original_flat > 0]
            
            imputed_flat = imputed_data.values.flatten()
            imputed_flat = imputed_flat[~np.isnan(imputed_flat)]
            
            ax.hist(np.log10(original_flat + 1), bins=50, alpha=0.6, 
                   label='Original', color='blue', edgecolor='black')
            ax.hist(np.log10(imputed_flat + 1), bins=50, alpha=0.6, 
                   label='After KNN', color='red', edgecolor='black')
            
            ax.set_xlabel('log10(Abundance + 1)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'Group: {group}', fontweight='bold', pad=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distribution_comparison.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_dir}/distribution_comparison.png")
        plt.close()
    
    def save_imputed_data(self, output_dir='knn_imputation_results', remove_empty_features=True):
        """Save imputed data to CSV"""
        Path(output_dir).mkdir(exist_ok=True)
        
        output_df = pd.concat([self.metadata, self.imputed_data], axis=1)
        
        if remove_empty_features:
            abundance_cols = self.imputed_data.columns
            features_with_data = output_df[abundance_cols].notna().any(axis=1)
            removed_count = (~features_with_data).sum()
            
            if removed_count > 0:
                print(f"\n! WARNING: Removing {removed_count} completely empty features")
                removed_features = output_df[~features_with_data]['Compound'].tolist()
                print(f"  Removed features: {removed_features[:10]}")
                if len(removed_features) > 10:
                    print(f"  ... and {len(removed_features) - 10} more")
            
            output_df = output_df[features_with_data]
        
        output_file = f'{output_dir}/imputed_data_knn.csv'
        output_df.to_csv(output_file, index=False)
        
        print(f"[OK] Saved imputed data: {output_file}")
        print(f"    Features: {len(output_df)}")
        print(f"    Samples: {len(self.imputed_data.columns)}")
        
        return output_df
    
    def generate_report(self, output_dir='knn_imputation_results'):
        """Generate comprehensive imputation report"""
        Path(output_dir).mkdir(exist_ok=True)
        
        report = []
        report.append("="*70)
        report.append("KNN IMPUTATION ANALYSIS REPORT")
        report.append("="*70)
        
        report.append("\nDataset Information:")
        report.append(f"  Total features: {len(self.abundance_data)}")
        report.append(f"  Total samples: {len(self.abundance_data.columns)}")
        report.append(f"  Biological groups: {list(self.groups.keys())}")
        
        abundance_cols = self.imputed_data.columns
        completely_missing = (~self.imputed_data.notna().any(axis=1)).sum()
        if completely_missing > 0:
            report.append(f"\n  ! {completely_missing} features were completely missing (no data to impute)")
        
        report.append("\nValidation Results (Cross-validation):")
        best_k = None
        best_rmse = float('inf')
        
        for k in sorted(self.validation_results.keys()):
            rmse = self.validation_results[k]['rmse_mean']
            if rmse < best_rmse:
                best_rmse = rmse
                best_k = k
            
            report.append(f"\n  k = {k}:")
            report.append(f"    RMSE: {self.validation_results[k]['rmse_mean']:.4f} +/- {self.validation_results[k]['rmse_std']:.4f}")
            report.append(f"    MAE:  {self.validation_results[k]['mae_mean']:.4f} +/- {self.validation_results[k]['mae_std']:.4f}")
            report.append(f"    R2:   {self.validation_results[k]['r2_mean']:.4f} +/- {self.validation_results[k]['r2_std']:.4f}")
            report.append(f"    Correlation: {self.validation_results[k]['correlation_mean']:.4f} +/- {self.validation_results[k]['correlation_std']:.4f}")
        
        report.append(f"\nRecommended k value: {best_k} (lowest RMSE)")
        
        report.append("\nInterpretation:")
        r2 = self.validation_results[best_k]['r2_mean']
        corr = self.validation_results[best_k]['correlation_mean']
        
        if r2 > 0.95 and corr > 0.98:
            report.append(f"  Excellent imputation quality (R2={r2:.3f}, Corr={corr:.3f})")
        elif r2 > 0.90 and corr > 0.95:
            report.append(f"  Good imputation quality (R2={r2:.3f}, Corr={corr:.3f})")
        elif r2 > 0.80 and corr > 0.90:
            report.append(f"  Acceptable imputation quality (R2={r2:.3f}, Corr={corr:.3f})")
        else:
            report.append(f"  Imputation quality needs review (R2={r2:.3f}, Corr={corr:.3f})")
        
        report.append("\n" + "="*70)
        
        report_text = "\n".join(report)
        print(report_text)
        
        with open(f'{output_dir}/knn_imputation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n[OK] Saved report: {output_dir}/knn_imputation_report.txt")


def main():
    """Main execution workflow"""
    csv_file = '../../data/progenesis_file.csv'
    
    print("="*70)
    print("KNN IMPUTATION ANALYSIS FOR METABOLOMICS")
    print("="*70)
    
    print(f"\nLoading data from: {csv_file}")
    loader = ProgenesisDataLoader(csv_file)
    metadata, abundance_data, groups = loader.load_data()
    
    print("\nInitializing KNN imputation analyzer...")
    analyzer = KNNImputationAnalyzer(abundance_data, groups, metadata)
    
    print("\nStep 1: Validation - Finding optimal k value")
    best_k = analyzer.validate_imputation(
        mask_fraction=0.1,
        k_values=[3, 5, 7, 10, 15],
        n_trials=5
    )
    
    print("\nStep 2: Plotting validation results")
    analyzer.plot_validation_results()
    
    print(f"\nStep 3: Performing final imputation with k={best_k}")
    stats = analyzer.impute_groupwise(k=best_k, scale=True)
    
    print("\nStep 4: Generating comparison plots")
    analyzer.plot_imputation_comparison()
    
    print("\nStep 5: Saving imputed data")
    analyzer.save_imputed_data()
    
    print("\nStep 6: Generating final report")
    analyzer.generate_report()
    
    print("\n" + "="*70)
    print("KNN IMPUTATION COMPLETE!")
    print("="*70)
    print("\nNext steps:")
    print("  1. Review validation metrics")
    print("  2. Compare with EM imputation")
    print("  3. Use imputed data for downstream analysis")


if __name__ == "__main__":
    main()