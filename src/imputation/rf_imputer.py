import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from visualize_missing import ProgenesisDataLoader


class RFImputationAnalyzer:
    """
    Random Forest-based imputation for metabolomics data
    Iterative approach: each feature predicted by all others using Random Forest
    """
    
    def __init__(self, abundance_data, groups, metadata=None):
        self.abundance_data = abundance_data
        self.groups = groups
        self.metadata = metadata
        self.imputed_data = None
        self.validation_results = {}
        self.convergence_history = {}
        
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
    
    def _initial_imputation(self, data):
        """Simple mean imputation as starting point"""
        data_filled = data.copy()
        
        for i in range(len(data)):
            row_mean = np.nanmean(data.iloc[i, :])
            if np.isnan(row_mean):
                row_mean = 0
            data_filled.iloc[i, :] = data_filled.iloc[i, :].fillna(row_mean)
        
        return data_filled
    
    def _rf_impute_iteration(self, data, data_filled, missing_mask, 
                             n_estimators=100, max_features='sqrt', 
                             verbose=False):
        """
        Single iteration of Random Forest imputation
        
        For each feature with missing values:
        1. Use other features as predictors
        2. Train Random Forest on complete cases
        3. Predict missing values
        """
        data_new = data_filled.copy()
        n_features = len(data)
        features_imputed = 0
        
        feature_order = np.random.permutation(n_features)
        
        for feat_idx in feature_order:
            missing_in_feat = missing_mask.iloc[feat_idx, :]
            
            if not missing_in_feat.any():
                continue
            
            other_features = [i for i in range(n_features) if i != feat_idx]
            
            train_mask = ~missing_in_feat
            if train_mask.sum() < 5:
                if verbose:
                    print(f"    Feature {feat_idx}: Too few training samples, skipping")
                continue
            
            train_cols = data_new.columns[train_mask]
            missing_cols = data_new.columns[missing_in_feat]
            
            X_train = data_new.loc[data_new.index[other_features], train_cols].T.values
            y_train = data_new.loc[data_new.index[feat_idx], train_cols].values
            
            X_test = data_new.loc[data_new.index[other_features], missing_cols].T.values
            
            if X_test.shape[0] == 0:
                continue
            
            try:
                rf = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_features=max_features,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                )
                
                rf.fit(X_train, y_train)
                
                y_pred = rf.predict(X_test)
                
                data_new.loc[data_new.index[feat_idx], missing_cols] = y_pred
                features_imputed += 1
                
            except Exception as e:
                if verbose:
                    print(f"    Feature {feat_idx}: Error during RF training - {e}")
                continue
        
        return data_new, features_imputed
    
    def _compute_imputation_error(self, data_original, data_imputed, missing_mask):
        """Compute RMSE between iterations on imputed values"""
        imputed_vals_original = data_original[missing_mask].values.flatten()
        imputed_vals_new = data_imputed[missing_mask].values.flatten()
        
        valid_mask = ~np.isnan(imputed_vals_original) & ~np.isnan(imputed_vals_new)
        
        if valid_mask.sum() == 0:
            return np.inf
        
        rmse = np.sqrt(np.mean((imputed_vals_original[valid_mask] - imputed_vals_new[valid_mask])**2))
        return rmse
    
    def impute_groupwise(self, max_iter=10, n_estimators=100, tol=1e-2, verbose=True):
        """
        Perform Random Forest imputation separately for each biological group
        
        Args:
            max_iter: maximum number of iterations
            n_estimators: number of trees in Random Forest
            tol: convergence tolerance for imputation change
            verbose: print progress
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"RANDOM FOREST IMPUTATION - GROUP-WISE")
            print(f"{'='*70}")
            print(f"  n_estimators: {n_estimators}")
            print(f"  max_iter: {max_iter}")
            print(f"  tolerance: {tol}")
        
        self.imputed_data = self.abundance_data.copy()
        self.convergence_history = {}
        
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
            
            features_with_data = group_data_clean.notna().any(axis=1)
            group_data_filtered = group_data_clean.loc[features_with_data]
            missing_mask_filtered = missing_mask.loc[features_with_data]
            
            if len(group_data_filtered) == 0:
                if verbose:
                    print(f"  No features with data in this group!")
                continue
            
            if verbose:
                print(f"  Data matrix shape: {group_data_filtered.shape}")
                print(f"  Running Random Forest imputation...")
            
            data_filled = self._initial_imputation(group_data_filtered)
            
            convergence_errors = []
            
            for iteration in range(max_iter):
                data_new, n_imputed = self._rf_impute_iteration(
                    group_data_filtered, 
                    data_filled, 
                    missing_mask_filtered,
                    n_estimators=n_estimators,
                    verbose=(verbose and iteration == 0)
                )
                
                error = self._compute_imputation_error(data_filled, data_new, missing_mask_filtered)
                convergence_errors.append(error)
                
                if iteration > 0:
                    relative_change = abs(convergence_errors[-1] - convergence_errors[-2]) / (convergence_errors[-2] + 1e-10)
                    
                    if verbose and iteration % 2 == 0:
                        print(f"    Iteration {iteration}: Error = {error:.2f}, Change = {relative_change:.6f}")
                    
                    if relative_change < tol:
                        if verbose:
                            print(f"    Converged at iteration {iteration} (change = {relative_change:.6f})")
                        break
                
                data_filled = data_new
            
            result_df = group_data_clean.copy()
            result_df.loc[features_with_data] = data_filled
            
            for col in cols:
                self.imputed_data[col] = result_df[col]
            
            self.convergence_history[group] = {
                'iterations': len(convergence_errors),
                'errors': convergence_errors
            }
            
            if verbose:
                print(f"  Final imputation error: {convergence_errors[-1]:.2f}")
                print(f"  Total iterations: {len(convergence_errors)}")
                print(f"  Imputed: {missing_before} values")
        
        if verbose:
            print(f"\n{'='*70}")
            print("RANDOM FOREST IMPUTATION COMPLETE")
            print(f"{'='*70}\n")
    
    def validate_imputation(self, mask_fraction=0.1, n_trials=5, max_iter=10, n_estimators=100):
        """
        Validate RF imputation by artificially masking known values
        """
        print(f"\n{'='*70}")
        print(f"RF VALIDATION: {n_trials} trials with {mask_fraction*100}% masking")
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
            
            temp_analyzer = RFImputationAnalyzer(masked_data, self.groups, self.metadata)
            temp_analyzer.impute_groupwise(max_iter=max_iter, n_estimators=n_estimators, 
                                          tol=1e-2, verbose=False)
            
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
    
    def plot_convergence(self, output_dir='rf_imputation_results'):
        """Plot convergence history for each group"""
        Path(output_dir).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (group, history) in enumerate(self.convergence_history.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            errors = history['errors']
            
            ax.plot(range(len(errors)), errors, marker='o', linewidth=2, markersize=6)
            ax.set_xlabel('Iteration', fontweight='bold')
            ax.set_ylabel('Imputation RMSE', fontweight='bold')
            ax.set_title(f'Group: {group}', fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3)
            
            if len(errors) > 1:
                final_error = errors[-1]
                ax.axhline(final_error, color='red', linestyle='--', alpha=0.7, 
                          label=f'Final: {final_error:.2f}')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/convergence_history.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_dir}/convergence_history.png")
        plt.close()
    
    def plot_imputation_comparison(self, output_dir='rf_imputation_results'):
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
                   label='After RF', color='orange', edgecolor='black')
            
            ax.set_xlabel('log10(Abundance + 1)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'Group: {group}', fontweight='bold', pad=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distribution_comparison.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_dir}/distribution_comparison.png")
        plt.close()
    
    def save_imputed_data(self, output_dir='rf_imputation_results', remove_empty_features=True):
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
        
        output_file = f'{output_dir}/imputed_data_rf.csv'
        output_df.to_csv(output_file, index=False)
        
        print(f"[OK] Saved imputed data: {output_file}")
        print(f"    Features: {len(output_df)}")
        print(f"    Samples: {len(self.imputed_data.columns)}")
        
        return output_df
    
    def generate_report(self, output_dir='rf_imputation_results'):
        """Generate comprehensive RF imputation report"""
        Path(output_dir).mkdir(exist_ok=True)
        
        report = []
        report.append("="*70)
        report.append("RANDOM FOREST IMPUTATION ANALYSIS REPORT")
        report.append("="*70)
        
        report.append("\nAlgorithm: Iterative Random Forest Imputation")
        report.append("\nDataset Information:")
        report.append(f"  Total features: {len(self.abundance_data)}")
        report.append(f"  Total samples: {len(self.abundance_data.columns)}")
        report.append(f"  Biological groups: {list(self.groups.keys())}")
        
        report.append("\nConvergence Information:")
        for group, history in self.convergence_history.items():
            report.append(f"  {group}:")
            report.append(f"    Iterations to convergence: {history['iterations']}")
            if len(history['errors']) > 0:
                report.append(f"    Initial error: {history['errors'][0]:.2f}")
                report.append(f"    Final error: {history['errors'][-1]:.2f}")
                if len(history['errors']) > 1:
                    error_reduction = history['errors'][0] - history['errors'][-1]
                    pct_reduction = (error_reduction / history['errors'][0]) * 100
                    report.append(f"    Error reduction: {error_reduction:.2f} ({pct_reduction:.1f}%)")
        
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
        
        report.append("\n" + "="*70)
        
        report_text = "\n".join(report)
        print(report_text)
        
        with open(f'{output_dir}/rf_imputation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n[OK] Saved report: {output_dir}/rf_imputation_report.txt")


def main():
    """Main execution workflow"""
    csv_file = '../../data/progenesis_file.csv'
    
    print("="*70)
    print("RANDOM FOREST IMPUTATION ANALYSIS FOR METABOLOMICS")
    print("="*70)
    
    print(f"\nLoading data from: {csv_file}")
    loader = ProgenesisDataLoader(csv_file)
    metadata, abundance_data, groups = loader.load_data()
    
    print("\nInitializing RF imputation analyzer...")
    analyzer = RFImputationAnalyzer(abundance_data, groups, metadata)
    
    print("\nStep 1: Validation - Testing RF imputation accuracy")
    analyzer.validate_imputation(mask_fraction=0.1, n_trials=5, max_iter=10, n_estimators=100)
    
    print("\nStep 2: Performing final RF imputation")
    analyzer.impute_groupwise(max_iter=10, n_estimators=100, tol=1e-2, verbose=True)
    
    print("\nStep 3: Plotting convergence history")
    analyzer.plot_convergence()
    
    print("\nStep 4: Generating comparison plots")
    analyzer.plot_imputation_comparison()
    
    print("\nStep 5: Saving imputed data")
    analyzer.save_imputed_data()
    
    print("\nStep 6: Generating final report")
    analyzer.generate_report()
    
    print("\n" + "="*70)
    print("RANDOM FOREST IMPUTATION COMPLETE!")
    print("="*70)
    print("\nNext step: Update comparison to include RF results")


if __name__ == "__main__":
    main()