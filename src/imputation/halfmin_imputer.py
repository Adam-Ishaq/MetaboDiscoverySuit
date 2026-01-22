import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from visualize_missing import ProgenesisDataLoader


class HalfMinImputationAnalyzer:
    """
    Half-Minimum imputation for metabolomics data
    
    Simple baseline approach commonly used in metabolomics:
    - Assumes missing values are below detection limit
    - Imputes with half of the minimum detected value for each feature
    - Fast and domain-specific, but ignores sample relationships
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
    
    def _prepare_data_for_imputation(self, data, replace_with_nan=True):
        """Convert data to numeric and replace missing values with NaN"""
        data_clean = data.copy()
        
        for col in data_clean.columns:
            data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')
        
        if replace_with_nan:
            data_clean = data_clean.replace(0, np.nan)
        
        return data_clean
    
    def impute_groupwise(self, global_min=False, verbose=True):
        """
        Perform Half-Minimum imputation separately for each biological group
        
        Args:
            global_min: If True, use global minimum across all samples.
                       If False, use group-specific minimum (default).
            verbose: print progress
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"HALF-MINIMUM IMPUTATION - GROUP-WISE")
            print(f"{'='*70}")
            print(f"  Strategy: {'Global minimum' if global_min else 'Group-specific minimum'}")
        
        self.imputed_data = self.abundance_data.copy()
        
        for group, cols in self.groups.items():
            if verbose:
                print(f"\nProcessing group: {group}")
                print(f"  Samples: {len(cols)}")
            
            group_data = self.abundance_data[cols].copy()
            group_data_clean = self._prepare_data_for_imputation(group_data)
            
            missing_mask = group_data_clean.isna()
            missing_before = missing_mask.sum().sum()
            total_values = group_data_clean.size
            missing_pct = (missing_before / total_values) * 100
            
            if verbose:
                print(f"  Missing values: {missing_before} ({missing_pct:.2f}%)")
            
            if missing_before == 0:
                if verbose:
                    print(f"  No missing values to impute!")
                for col in cols:
                    self.imputed_data[col] = group_data_clean[col]
                continue
            
            imputed_data = group_data_clean.copy()
            
            for feature_idx in group_data_clean.index:
                feature_values = group_data_clean.loc[feature_idx, :]
                
                if feature_values.isna().any():
                    min_value = feature_values.min()
                    
                    if pd.isna(min_value) or min_value <= 0:
                        if global_min:
                            overall_min = self.abundance_data.min().min()
                            if overall_min > 0:
                                impute_value = overall_min / 2
                            else:
                                impute_value = 1.0
                        else:
                            impute_value = 1.0
                    else:
                        impute_value = min_value / 2
                    
                    missing_indices = feature_values.isna()
                    imputed_data.loc[feature_idx, missing_indices] = impute_value
            
            for col in cols:
                self.imputed_data[col] = imputed_data[col]
            
            imputed_values = imputed_data[missing_mask].values.flatten()
            imputed_values = imputed_values[~np.isnan(imputed_values)]
            unique_imputed_values = len(np.unique(imputed_values))
            
            if verbose:
                print(f"  Imputed: {missing_before} values")
                print(f"  Unique imputed values: {unique_imputed_values}")
        
        if verbose:
            print(f"\n{'='*70}")
            print("HALF-MINIMUM IMPUTATION COMPLETE")
            print(f"{'='*70}\n")
    
    def validate_imputation(self, mask_fraction=0.1, n_trials=5, global_min=False):
        """
        Validate Half-Min imputation by artificially masking known values
        """
        print(f"\n{'='*70}")
        print(f"HALF-MIN VALIDATION: {n_trials} trials with {mask_fraction*100}% masking")
        print(f"{'='*70}")
        
        validation_metrics = {'rmse': [], 'mae': [], 'r2': [], 'correlation': []}
        
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
            
            temp_analyzer = HalfMinImputationAnalyzer(masked_data, self.groups, self.metadata)
            temp_analyzer.impute_groupwise(global_min=global_min, verbose=False)
            
            predicted_values = []
            for r, c in zip(rows_to_mask, cols_to_mask):
                predicted_values.append(temp_analyzer.imputed_data.iloc[r, c])
            predicted_values = np.array(predicted_values)
            
            valid_mask = ~np.isnan(predicted_values) & ~np.isnan(true_values)
            
            if np.sum(valid_mask) < len(true_values) * 0.5:
                print(f"  WARNING: Only {np.sum(valid_mask)}/{len(true_values)} valid predictions")
                continue
            
            true_values_valid = true_values[valid_mask]
            predicted_values_valid = predicted_values[valid_mask]
            
            if len(true_values_valid) == 0:
                print(f"  ERROR: No valid predictions to evaluate!")
                continue
            
            rmse = np.sqrt(mean_squared_error(true_values_valid, predicted_values_valid))
            mae = mean_absolute_error(true_values_valid, predicted_values_valid)
            r2 = r2_score(true_values_valid, predicted_values_valid)
            corr, _ = pearsonr(true_values_valid, predicted_values_valid)
            
            validation_metrics['rmse'].append(rmse)
            validation_metrics['mae'].append(mae)
            validation_metrics['r2'].append(r2)
            validation_metrics['correlation'].append(corr)
            
            print(f"  RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, Corr: {corr:.4f}")
        
        self.validation_results = {
            'rmse_mean': np.mean(validation_metrics['rmse']),
            'rmse_std': np.std(validation_metrics['rmse']),
            'mae_mean': np.mean(validation_metrics['mae']),
            'mae_std': np.std(validation_metrics['mae']),
            'r2_mean': np.mean(validation_metrics['r2']),
            'r2_std': np.std(validation_metrics['r2']),
            'correlation_mean': np.mean(validation_metrics['correlation']),
            'correlation_std': np.std(validation_metrics['correlation'])
        }
        
        print(f"\n{'='*70}")
        print("VALIDATION RESULTS")
        print(f"{'='*70}")
        print(f"RMSE: {self.validation_results['rmse_mean']:.4f} +/- {self.validation_results['rmse_std']:.4f}")
        print(f"MAE:  {self.validation_results['mae_mean']:.4f} +/- {self.validation_results['mae_std']:.4f}")
        print(f"R2:   {self.validation_results['r2_mean']:.4f} +/- {self.validation_results['r2_std']:.4f}")
        print(f"Corr: {self.validation_results['correlation_mean']:.4f} +/- {self.validation_results['correlation_std']:.4f}")
        print(f"{'='*70}\n")
    
    def plot_imputation_comparison(self, output_dir='halfmin_imputation_results'):
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
                   label='After Half-Min', color='red', edgecolor='black')
            
            ax.set_xlabel('log10(Abundance + 1)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'Group: {group}', fontweight='bold', pad=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distribution_comparison.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_dir}/distribution_comparison.png")
        plt.close()
    
    def save_imputed_data(self, output_dir='halfmin_imputation_results', remove_empty_features=True):
        """Save imputed data to CSV"""
        Path(output_dir).mkdir(exist_ok=True)
        
        output_df = pd.concat([self.metadata, self.imputed_data], axis=1)
        
        if remove_empty_features:
            abundance_cols = self.imputed_data.columns
            features_with_data = output_df[abundance_cols].notna().any(axis=1)
            removed_count = (~features_with_data).sum()
            
            if removed_count > 0:
                print(f"\n! WARNING: Removing {removed_count} completely empty features")
            
            output_df = output_df[features_with_data]
        
        output_file = f'{output_dir}/imputed_data_halfmin.csv'
        output_df.to_csv(output_file, index=False)
        
        print(f"[OK] Saved imputed data: {output_file}")
        print(f"    Features: {len(output_df)}")
        print(f"    Samples: {len(self.imputed_data.columns)}")
        
        return output_df
    
    def generate_report(self, output_dir='halfmin_imputation_results'):
        """Generate comprehensive Half-Min imputation report"""
        Path(output_dir).mkdir(exist_ok=True)
        
        report = []
        report.append("="*70)
        report.append("HALF-MINIMUM IMPUTATION ANALYSIS REPORT")
        report.append("="*70)
        
        report.append("\nAlgorithm: Half-Minimum (Domain-Specific Baseline)")
        report.append("\nMethod Description:")
        report.append("  - Assumes missing values are below detection limit")
        report.append("  - Imputes with 50% of minimum detected value per feature")
        report.append("  - Simple, fast, domain-specific approach")
        
        report.append("\nDataset Information:")
        report.append(f"  Total features: {len(self.abundance_data)}")
        report.append(f"  Total samples: {len(self.abundance_data.columns)}")
        report.append(f"  Biological groups: {list(self.groups.keys())}")
        
        if self.validation_results:
            report.append("\nValidation Results (Cross-validation):")
            report.append(f"  RMSE: {self.validation_results['rmse_mean']:.4f} +/- {self.validation_results['rmse_std']:.4f}")
            report.append(f"  MAE:  {self.validation_results['mae_mean']:.4f} +/- {self.validation_results['mae_std']:.4f}")
            report.append(f"  R2:   {self.validation_results['r2_mean']:.4f} +/- {self.validation_results['r2_std']:.4f}")
            report.append(f"  Correlation: {self.validation_results['correlation_mean']:.4f} +/- {self.validation_results['correlation_std']:.4f}")
            
            report.append("\nInterpretation:")
            r2 = self.validation_results['r2_mean']
            corr = self.validation_results['correlation_mean']
            
            if r2 > 0.95 and corr > 0.98:
                report.append(f"  Excellent imputation quality (R2={r2:.3f}, Corr={corr:.3f})")
            elif r2 > 0.90 and corr > 0.95:
                report.append(f"  Good imputation quality (R2={r2:.3f}, Corr={corr:.3f})")
            elif r2 > 0.80 and corr > 0.90:
                report.append(f"  Acceptable imputation quality (R2={r2:.3f}, Corr={corr:.3f})")
            else:
                report.append(f"  Imputation quality needs review (R2={r2:.3f}, Corr={corr:.3f})")
            
            report.append("\nNote:")
            report.append("  Half-minimum is a simple baseline approach.")
            report.append("  It does not account for:")
            report.append("    - Sample similarity (unlike KNN)")
            report.append("    - Correlation structure (unlike EM/SVD)")
            report.append("    - Non-linear relationships (unlike RF)")
            report.append("  However, it is:")
            report.append("    - Fast and computationally efficient")
            report.append("    - Domain-specific for metabolomics")
            report.append("    - Conservative (doesn't overestimate low values)")
        
        report.append("\n" + "="*70)
        
        report_text = "\n".join(report)
        print(report_text)
        
        with open(f'{output_dir}/halfmin_imputation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n[OK] Saved report: {output_dir}/halfmin_imputation_report.txt")


def main():
    """Main execution workflow"""
    csv_file = '../../data/progenesis_file.csv'
    
    print("="*70)
    print("HALF-MINIMUM IMPUTATION ANALYSIS FOR METABOLOMICS")
    print("="*70)
    
    print(f"\nLoading data from: {csv_file}")
    loader = ProgenesisDataLoader(csv_file)
    metadata, abundance_data, groups = loader.load_data()
    
    print("\nInitializing Half-Min imputation analyzer...")
    analyzer = HalfMinImputationAnalyzer(abundance_data, groups, metadata)
    
    print("\nStep 1: Validation - Testing Half-Min imputation accuracy")
    analyzer.validate_imputation(mask_fraction=0.1, n_trials=5, global_min=False)
    
    print("\nStep 2: Performing final Half-Min imputation")
    analyzer.impute_groupwise(global_min=False, verbose=True)
    
    print("\nStep 3: Generating comparison plots")
    analyzer.plot_imputation_comparison()
    
    print("\nStep 4: Saving imputed data")
    analyzer.save_imputed_data()
    
    print("\nStep 5: Generating final report")
    analyzer.generate_report()
    
    print("\n" + "="*70)
    print("HALF-MINIMUM IMPUTATION COMPLETE!")
    print("="*70)
    print("\nNext step: Update comparison to include Half-Min results")


if __name__ == "__main__":
    main()