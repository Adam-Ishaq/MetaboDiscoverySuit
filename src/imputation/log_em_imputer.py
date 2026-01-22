import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr
from scipy import linalg
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from visualize_missing import ProgenesisDataLoader


class LogNormalEMImputationAnalyzer:
    """
    Log-Normal EM imputation for metabolomics data
    
    Key Innovation:
    - Metabolomics data is typically log-normally distributed
    - Apply log transformation before EM
    - Run EM assuming Gaussian on log-scale
    - Back-transform to original scale
    
    This respects the natural distribution of metabolite abundances!
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
    
    def _log_transform(self, data, offset=1.0):
        """Apply log transformation with small offset to handle zeros"""
        return np.log(data + offset)
    
    def _inverse_log_transform(self, data, offset=1.0):
        """Inverse log transformation"""
        return np.exp(data) - offset
    
    def _initialize_parameters(self, data):
        """Initialize mean and covariance on log-scale"""
        n_features = data.shape[0]
        
        mu = np.nanmean(data.values, axis=1)
        
        data_centered = data.values - mu[:, np.newaxis]
        mask = ~np.isnan(data_centered)
        
        sigma = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i, n_features):
                valid_idx = mask[i, :] & mask[j, :]
                if np.sum(valid_idx) > 1:
                    sigma[i, j] = np.nanmean(data_centered[i, valid_idx] * data_centered[j, valid_idx])
                    sigma[j, i] = sigma[i, j]
        
        lambda_reg = 0.01 * np.mean(np.diag(sigma))
        sigma += lambda_reg * np.eye(n_features)
        
        min_eigenval = np.min(np.linalg.eigvalsh(sigma))
        if min_eigenval < 1e-6:
            sigma += (1e-6 - min_eigenval) * np.eye(n_features)
        
        return mu, sigma
    
    def _conditional_distribution(self, x_obs, obs_idx, miss_idx, mu, sigma):
        """Compute conditional distribution on log-scale"""
        if len(miss_idx) == 0:
            return np.array([]), np.array([[]])
        
        mu_obs = mu[obs_idx]
        mu_miss = mu[miss_idx]
        
        sigma_11 = sigma[np.ix_(obs_idx, obs_idx)]
        sigma_12 = sigma[np.ix_(obs_idx, miss_idx)]
        sigma_21 = sigma[np.ix_(miss_idx, obs_idx)]
        sigma_22 = sigma[np.ix_(miss_idx, miss_idx)]
        
        sigma_11_inv = np.linalg.pinv(sigma_11, rcond=1e-6)
        
        mu_cond = mu_miss + sigma_21 @ sigma_11_inv @ (x_obs - mu_obs)
        sigma_cond = sigma_22 - sigma_21 @ sigma_11_inv @ sigma_12
        
        min_eigenval = np.min(np.linalg.eigvalsh(sigma_cond))
        if min_eigenval < 1e-8:
            sigma_cond += (1e-8 - min_eigenval) * np.eye(len(miss_idx))
        
        return mu_cond, sigma_cond
    
    def _compute_log_likelihood(self, data, mu, sigma, verbose=False):
        """Compute log-likelihood on log-scale"""
        n_samples = data.shape[1]
        log_likelihood = 0.0
        n_computed = 0
        
        for j in range(n_samples):
            x_j = data[:, j]
            obs_idx = np.where(~np.isnan(x_j))[0]
            
            if len(obs_idx) == 0:
                continue
            
            x_obs = x_j[obs_idx]
            mu_obs = mu[obs_idx]
            sigma_obs = sigma[np.ix_(obs_idx, obs_idx)]
            
            min_eigenval = np.min(np.linalg.eigvalsh(sigma_obs))
            if min_eigenval < 1e-8:
                sigma_obs = sigma_obs + (1e-8 - min_eigenval) * np.eye(len(obs_idx))
            
            try:
                sign, log_det = np.linalg.slogdet(sigma_obs)
                if sign <= 0:
                    if verbose:
                        print(f"  WARNING: Non-positive definite covariance for sample {j}")
                    continue
                
                sigma_obs_inv = np.linalg.pinv(sigma_obs, rcond=1e-8)
                diff = x_obs - mu_obs
                mahal = diff.T @ sigma_obs_inv @ diff
                
                sample_ll = -0.5 * (len(obs_idx) * np.log(2 * np.pi) + log_det + mahal)
                
                if not np.isfinite(sample_ll):
                    if verbose:
                        print(f"  WARNING: Non-finite log-likelihood for sample {j}: {sample_ll}")
                    continue
                
                log_likelihood += sample_ll
                n_computed += 1
                
            except np.linalg.LinAlgError as e:
                if verbose:
                    print(f"  WARNING: LinAlgError for sample {j}: {e}")
                continue
        
        if n_computed == 0:
            if verbose:
                print("  WARNING: No valid log-likelihood computations!")
            return -np.inf
        
        return log_likelihood
    
    def _em_iteration(self, data, mu, sigma, verbose=False):
        """Single EM iteration on log-scale"""
        n_features, n_samples = data.shape
        
        x_filled = data.copy()
        sigma_sum = np.zeros((n_features, n_features))
        n_imputed = 0
        
        for j in range(n_samples):
            x_j = data[:, j]
            obs_idx = np.where(~np.isnan(x_j))[0]
            miss_idx = np.where(np.isnan(x_j))[0]
            
            if len(miss_idx) > 0 and len(obs_idx) > 0:
                x_obs = x_j[obs_idx]
                
                mu_cond, sigma_cond = self._conditional_distribution(
                    x_obs, obs_idx, miss_idx, mu, sigma
                )
                
                x_filled[miss_idx, j] = mu_cond
                
                sigma_contrib = np.zeros((n_features, n_features))
                sigma_contrib[np.ix_(miss_idx, miss_idx)] = sigma_cond
                sigma_sum += sigma_contrib
                n_imputed += len(miss_idx)
        
        mu_new = np.nanmean(x_filled, axis=1)
        
        x_centered = x_filled - mu_new[:, np.newaxis]
        sigma_new = (x_centered @ x_centered.T) / n_samples + sigma_sum / n_samples
        
        lambda_reg = 0.001 * np.mean(np.diag(sigma_new))
        sigma_new += lambda_reg * np.eye(n_features)
        
        min_eigenval = np.min(np.linalg.eigvalsh(sigma_new))
        if min_eigenval < 1e-6:
            sigma_new += (1e-6 - min_eigenval) * np.eye(n_features)
        
        log_likelihood = self._compute_log_likelihood(data, mu_new, sigma_new, verbose=verbose)
        
        return mu_new, sigma_new, log_likelihood, x_filled
    
    def impute_groupwise(self, max_iter=100, tol=1e-3, verbose=True):
        """
        Perform Log-Normal EM imputation for each group
        
        Process:
        1. Log-transform data
        2. Run EM on log-scale (Gaussian assumption)
        3. Back-transform to original scale
        """
        if verbose:
            print(f"\n{'='*70}")
            print(f"LOG-NORMAL EM IMPUTATION - GROUP-WISE")
            print(f"{'='*70}")
            print(f"  Innovation: Log-transform → EM → Back-transform")
            print(f"  Respects log-normal distribution of metabolomics data")
        
        self.imputed_data = self.abundance_data.copy()
        self.convergence_history = {}
        
        for group, cols in self.groups.items():
            if verbose:
                print(f"\nProcessing group: {group}")
                print(f"  Samples: {len(cols)}")
            
            group_data = self.abundance_data[cols].copy()
            group_data_clean = self._prepare_data_for_imputation(group_data)
            
            missing_before = group_data_clean.isna().sum().sum()
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
            
            if len(group_data_filtered) == 0:
                if verbose:
                    print(f"  No features with data in this group!")
                continue
            
            offset = np.nanmin(group_data_filtered.values) / 2
            if offset <= 0 or np.isnan(offset):
                offset = 1.0
            
            if verbose:
                print(f"  Log-transform offset: {offset:.4f}")
            
            group_data_log = self._log_transform(group_data_filtered, offset)
            
            data_matrix = group_data_log.values
            
            if verbose:
                print(f"  Data matrix shape: {data_matrix.shape}")
                print(f"  Running EM on log-scale...")
            
            mu, sigma = self._initialize_parameters(group_data_log)
            
            if verbose:
                print(f"  Initial mean range (log): [{np.min(mu):.2f}, {np.max(mu):.2f}]")
            
            log_likelihoods = []
            initial_ll = self._compute_log_likelihood(data_matrix, mu, sigma, verbose=verbose)
            log_likelihoods.append(initial_ll)
            
            if verbose:
                print(f"    Initial log-likelihood: {initial_ll:.2f}")
            
            for iteration in range(max_iter):
                mu_new, sigma_new, log_likelihood, x_filled = self._em_iteration(
                    data_matrix, mu, sigma, verbose=(verbose and iteration == 0)
                )
                
                log_likelihoods.append(log_likelihood)
                
                if iteration > 0:
                    ll_change = abs(log_likelihood - log_likelihoods[-2])
                    
                    if iteration % 10 == 0 and verbose:
                        print(f"    Iteration {iteration}: LL = {log_likelihood:.2f}, Change = {ll_change:.6f}")
                    
                    if ll_change < tol:
                        if verbose:
                            print(f"    Converged at iteration {iteration} (LL change = {ll_change:.6f})")
                        break
                
                mu = mu_new
                sigma = sigma_new
            
            imputed_df_log = pd.DataFrame(
                x_filled,
                index=group_data_filtered.index,
                columns=group_data_filtered.columns
            )
            
            imputed_df_original = self._inverse_log_transform(imputed_df_log, offset)
            
            imputed_df_original = imputed_df_original.clip(lower=0)
            
            result_df = group_data_clean.copy()
            result_df.loc[features_with_data] = imputed_df_original
            
            for col in cols:
                self.imputed_data[col] = result_df[col]
            
            self.convergence_history[group] = {
                'iterations': len(log_likelihoods) - 1,
                'log_likelihoods': log_likelihoods,
                'final_mu': mu,
                'final_sigma': sigma,
                'log_offset': offset
            }
            
            if verbose:
                print(f"  Final log-likelihood: {log_likelihoods[-1]:.2f}")
                print(f"  Total iterations: {len(log_likelihoods) - 1}")
                print(f"  Imputed: {missing_before} values")
                print(f"  Back-transformed to original scale")
        
        if verbose:
            print(f"\n{'='*70}")
            print("LOG-NORMAL EM IMPUTATION COMPLETE")
            print(f"{'='*70}\n")
    
    def validate_imputation(self, mask_fraction=0.1, n_trials=5, max_iter=50):
        """Validate Log-Normal EM imputation"""
        print(f"\n{'='*70}")
        print(f"LOG-NORMAL EM VALIDATION: {n_trials} trials with {mask_fraction*100}% masking")
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
            
            temp_analyzer = LogNormalEMImputationAnalyzer(masked_data, self.groups, self.metadata)
            temp_analyzer.impute_groupwise(max_iter=max_iter, tol=1e-3, verbose=False)
            
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
    
    def plot_convergence(self, output_dir='lognormal_em_imputation_results'):
        """Plot convergence history"""
        Path(output_dir).mkdir(exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, (group, history) in enumerate(self.convergence_history.items()):
            if idx >= 4:
                break
            
            ax = axes[idx]
            log_lls = history['log_likelihoods']
            
            ax.plot(range(len(log_lls)), log_lls, marker='o', linewidth=2, markersize=6, color='darkblue')
            ax.set_xlabel('Iteration', fontweight='bold')
            ax.set_ylabel('Log-Likelihood', fontweight='bold')
            ax.set_title(f'Group: {group}', fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3)
            
            if len(log_lls) > 1:
                final_ll = log_lls[-1]
                ax.axhline(final_ll, color='red', linestyle='--', alpha=0.7, 
                          label=f'Final: {final_ll:.2f}')
                ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/convergence_history.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_dir}/convergence_history.png")
        plt.close()
    
    def plot_imputation_comparison(self, output_dir='lognormal_em_imputation_results'):
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
                   label='After Log-Normal EM', color='darkblue', edgecolor='black')
            
            ax.set_xlabel('log10(Abundance + 1)', fontweight='bold')
            ax.set_ylabel('Frequency', fontweight='bold')
            ax.set_title(f'Group: {group}', fontweight='bold', pad=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distribution_comparison.png', dpi=300, bbox_inches='tight')
        print(f"[OK] Saved: {output_dir}/distribution_comparison.png")
        plt.close()
    
    def save_imputed_data(self, output_dir='lognormal_em_imputation_results', remove_empty_features=True):
        """Save imputed data"""
        Path(output_dir).mkdir(exist_ok=True)
        
        output_df = pd.concat([self.metadata, self.imputed_data], axis=1)
        
        if remove_empty_features:
            abundance_cols = self.imputed_data.columns
            features_with_data = output_df[abundance_cols].notna().any(axis=1)
            removed_count = (~features_with_data).sum()
            
            if removed_count > 0:
                print(f"\n! WARNING: Removing {removed_count} completely empty features")
            
            output_df = output_df[features_with_data]
        
        output_file = f'{output_dir}/imputed_data_lognormal_em.csv'
        output_df.to_csv(output_file, index=False)
        
        print(f"[OK] Saved imputed data: {output_file}")
        print(f"    Features: {len(output_df)}")
        print(f"    Samples: {len(self.imputed_data.columns)}")
        
        return output_df
    
    def generate_report(self, output_dir='lognormal_em_imputation_results'):
        """Generate report"""
        Path(output_dir).mkdir(exist_ok=True)
        
        report = []
        report.append("="*70)
        report.append("LOG-NORMAL EM IMPUTATION ANALYSIS REPORT")
        report.append("="*70)
        
        report.append("\nAlgorithm: Log-Normal EM (Distribution-Aware)")
        report.append("\nKey Innovation:")
        report.append("  - Log-transform data before EM")
        report.append("  - Apply Gaussian EM on log-scale")
        report.append("  - Back-transform to original scale")
        report.append("  - Respects log-normal distribution of metabolomics data")
        
        report.append("\nDataset Information:")
        report.append(f"  Total features: {len(self.abundance_data)}")
        report.append(f"  Total samples: {len(self.abundance_data.columns)}")
        report.append(f"  Biological groups: {list(self.groups.keys())}")
        
        report.append("\nConvergence Information:")
        for group, history in self.convergence_history.items():
            report.append(f"  {group}:")
            report.append(f"    Iterations: {history['iterations']}")
            if len(history['log_likelihoods']) > 0:
                report.append(f"    Initial LL: {history['log_likelihoods'][0]:.2f}")
                report.append(f"    Final LL: {history['log_likelihoods'][-1]:.2f}")
                if len(history['log_likelihoods']) > 1:
                    ll_improvement = history['log_likelihoods'][-1] - history['log_likelihoods'][0]
                    report.append(f"    LL improvement: {ll_improvement:.2f}")
            report.append(f"    Log offset used: {history['log_offset']:.4f}")
        
        if self.validation_results:
            report.append("\nValidation Results:")
            report.append(f"  RMSE: {self.validation_results['rmse_mean']:.4f} +/- {self.validation_results['rmse_std']:.4f}")
            report.append(f"  MAE:  {self.validation_results['mae_mean']:.4f} +/- {self.validation_results['mae_std']:.4f}")
            report.append(f"  R2:   {self.validation_results['r2_mean']:.4f} +/- {self.validation_results['r2_std']:.4f}")
            report.append(f"  Corr: {self.validation_results['correlation_mean']:.4f} +/- {self.validation_results['correlation_std']:.4f}")
            
            report.append("\nInterpretation:")
            r2 = self.validation_results['r2_mean']
            corr = self.validation_results['correlation_mean']
            
            if r2 > 0.95 and corr > 0.98:
                report.append(f"  Excellent (R2={r2:.3f}, Corr={corr:.3f})")
            elif r2 > 0.90 and corr > 0.95:
                report.append(f"  Good (R2={r2:.3f}, Corr={corr:.3f})")
            else:
                report.append(f"  Acceptable (R2={r2:.3f}, Corr={corr:.3f})")
        
        report.append("\n" + "="*70)
        
        report_text = "\n".join(report)
        print(report_text)
        
        with open(f'{output_dir}/lognormal_em_imputation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n[OK] Saved: {output_dir}/lognormal_em_imputation_report.txt")


def main():
    """Main execution"""
    csv_file = '../../data/progenesis_file.csv'
    
    print("="*70)
    print("LOG-NORMAL EM IMPUTATION FOR METABOLOMICS")
    print("="*70)
    
    print(f"\nLoading data from: {csv_file}")
    loader = ProgenesisDataLoader(csv_file)
    metadata, abundance_data, groups = loader.load_data()
    
    print("\nInitializing Log-Normal EM analyzer...")
    analyzer = LogNormalEMImputationAnalyzer(abundance_data, groups, metadata)
    
    print("\nStep 1: Validation")
    analyzer.validate_imputation(mask_fraction=0.1, n_trials=5, max_iter=50)
    
    print("\nStep 2: Final imputation")
    analyzer.impute_groupwise(max_iter=100, tol=1e-3, verbose=True)
    
    print("\nStep 3: Plotting convergence")
    analyzer.plot_convergence()
    
    print("\nStep 4: Comparison plots")
    analyzer.plot_imputation_comparison()
    
    print("\nStep 5: Saving data")
    analyzer.save_imputed_data()
    
    print("\nStep 6: Report")
    analyzer.generate_report()
    
    print("\n" + "="*70)
    print("LOG-NORMAL EM COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()