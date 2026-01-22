import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append('..')
from visualize_missing import ProgenesisDataLoader


class ImputationComparator:
    """
    Comprehensive comparison framework for KNN vs EM vs RF vs SVD vs Half-Min imputation methods
    Evaluates all methods across multiple dimensions: accuracy, distribution, biological validity
    """
    
    def __init__(self, original_data, knn_data, em_data, rf_data, svd_data, halfmin_data, groups, metadata):
        self.original_data = original_data
        self.knn_data = knn_data
        self.em_data = em_data
        self.rf_data = rf_data
        self.svd_data = svd_data
        self.halfmin_data = halfmin_data
        self.groups = groups
        self.metadata = metadata
        self.comparison_results = {}
        
        sns.set_style("whitegrid")
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
    
    def _identify_imputed_values(self, original, imputed):
        """Identify which values were imputed (originally missing)"""
        original_numeric = original.copy()
        for col in original_numeric.columns:
            original_numeric[col] = pd.to_numeric(original_numeric[col], errors='coerce')
        
        original_numeric = original_numeric.replace(0, np.nan)
        
        imputed_mask = original_numeric.isna() & imputed.notna()
        return imputed_mask
    
    def compare_imputed_distributions(self, output_dir='comparison_results'):
        """Compare the distributions of imputed values between all methods"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("DISTRIBUTION COMPARISON: KNN vs EM vs RF vs SVD vs Half-Min")
        print("="*70)
        
        imputed_mask = self._identify_imputed_values(self.original_data, self.knn_data)
        
        knn_imputed_vals = self.knn_data[imputed_mask].values.flatten()
        knn_imputed_vals = knn_imputed_vals[~np.isnan(knn_imputed_vals)]
        
        em_imputed_vals = self.em_data[imputed_mask].values.flatten()
        em_imputed_vals = em_imputed_vals[~np.isnan(em_imputed_vals)]
        
        rf_imputed_vals = self.rf_data[imputed_mask].values.flatten()
        rf_imputed_vals = rf_imputed_vals[~np.isnan(rf_imputed_vals)]
        
        svd_imputed_vals = self.svd_data[imputed_mask].values.flatten()
        svd_imputed_vals = svd_imputed_vals[~np.isnan(svd_imputed_vals)]
        
        halfmin_imputed_vals = self.halfmin_data[imputed_mask].values.flatten()
        halfmin_imputed_vals = halfmin_imputed_vals[~np.isnan(halfmin_imputed_vals)]
        
        print(f"\nImputed values statistics:")
        print(f"  KNN:     n={len(knn_imputed_vals)}, mean={np.mean(knn_imputed_vals):.2f}, std={np.std(knn_imputed_vals):.2f}")
        print(f"  EM:      n={len(em_imputed_vals)}, mean={np.mean(em_imputed_vals):.2f}, std={np.std(em_imputed_vals):.2f}")
        print(f"  RF:      n={len(rf_imputed_vals)}, mean={np.mean(rf_imputed_vals):.2f}, std={np.std(rf_imputed_vals):.2f}")
        print(f"  SVD:     n={len(svd_imputed_vals)}, mean={np.mean(svd_imputed_vals):.2f}, std={np.std(svd_imputed_vals):.2f}")
        print(f"  HalfMin: n={len(halfmin_imputed_vals)}, mean={np.mean(halfmin_imputed_vals):.2f}, std={np.std(halfmin_imputed_vals):.2f}")
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax = axes[0, 0]
        ax.hist(np.log10(knn_imputed_vals + 1), bins=50, alpha=0.3, 
               label='KNN', color='blue', edgecolor='black', density=True)
        ax.hist(np.log10(em_imputed_vals + 1), bins=50, alpha=0.3, 
               label='EM', color='green', edgecolor='black', density=True)
        ax.hist(np.log10(rf_imputed_vals + 1), bins=50, alpha=0.3, 
               label='RF', color='orange', edgecolor='black', density=True)
        ax.hist(np.log10(svd_imputed_vals + 1), bins=50, alpha=0.3, 
               label='SVD', color='purple', edgecolor='black', density=True)
        ax.hist(np.log10(halfmin_imputed_vals + 1), bins=50, alpha=0.3, 
               label='Half-Min', color='red', edgecolor='black', density=True)
        ax.set_xlabel('log10(Imputed Values + 1)', fontweight='bold')
        ax.set_ylabel('Density', fontweight='bold')
        ax.set_title('Distribution of Imputed Values', fontweight='bold', pad=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        ax.boxplot([np.log10(knn_imputed_vals + 1), np.log10(em_imputed_vals + 1), 
                   np.log10(rf_imputed_vals + 1), np.log10(svd_imputed_vals + 1),
                   np.log10(halfmin_imputed_vals + 1)],
                  labels=['KNN', 'EM', 'RF', 'SVD', 'HM'])
        ax.set_ylabel('log10(Imputed Values + 1)', fontweight='bold')
        ax.set_title('Boxplot Comparison', fontweight='bold', pad=10)
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        stats_df = pd.DataFrame({
            'Method': ['KNN', 'EM', 'RF', 'SVD', 'HM'],
            'Mean': [np.mean(knn_imputed_vals), np.mean(em_imputed_vals), 
                    np.mean(rf_imputed_vals), np.mean(svd_imputed_vals),
                    np.mean(halfmin_imputed_vals)],
            'Median': [np.median(knn_imputed_vals), np.median(em_imputed_vals), 
                      np.median(rf_imputed_vals), np.median(svd_imputed_vals),
                      np.median(halfmin_imputed_vals)],
            'Std': [np.std(knn_imputed_vals), np.std(em_imputed_vals), 
                   np.std(rf_imputed_vals), np.std(svd_imputed_vals),
                   np.std(halfmin_imputed_vals)]
        })
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=stats_df.values, colLabels=stats_df.columns,
                        cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        ax.set_title('Statistical Summary', fontweight='bold', pad=20)
        
        ax = axes[1, 1]
        sample_size = min(5000, len(knn_imputed_vals), len(halfmin_imputed_vals))
        knn_sample = np.random.choice(knn_imputed_vals, sample_size, replace=False)
        halfmin_sample = np.random.choice(halfmin_imputed_vals, sample_size, replace=False)
        ax.scatter(np.log10(knn_sample + 1), np.log10(halfmin_sample + 1), 
                  alpha=0.3, s=10, color='red')
        lims = [ax.get_xlim(), ax.get_ylim()]
        lims = [min(lims[0][0], lims[1][0]), max(lims[0][1], lims[1][1])]
        ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=2, label='y=x')
        ax.set_xlabel('log10(KNN + 1)', fontweight='bold')
        ax.set_ylabel('log10(Half-Min + 1)', fontweight='bold')
        ax.set_title('KNN vs Half-Min', fontweight='bold', pad=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/distribution_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved: {output_dir}/distribution_comparison.png")
        plt.close()
        
        self.comparison_results['distribution'] = {
            'knn_mean': np.mean(knn_imputed_vals),
            'em_mean': np.mean(em_imputed_vals),
            'rf_mean': np.mean(rf_imputed_vals),
            'svd_mean': np.mean(svd_imputed_vals),
            'halfmin_mean': np.mean(halfmin_imputed_vals),
            'knn_std': np.std(knn_imputed_vals),
            'em_std': np.std(em_imputed_vals),
            'rf_std': np.std(rf_imputed_vals),
            'svd_std': np.std(svd_imputed_vals),
            'halfmin_std': np.std(halfmin_imputed_vals)
        }
    
    def compare_group_separation(self, output_dir='comparison_results'):
        """Compare how well each method preserves biological group separation using PCA"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("GROUP SEPARATION COMPARISON: PCA Analysis")
        print("="*70)
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        for idx, (method_name, data) in enumerate([('KNN', self.knn_data), ('EM', self.em_data), 
                                                     ('RF', self.rf_data), ('SVD', self.svd_data),
                                                     ('Half-Min', self.halfmin_data)]):
            data_clean = data.dropna()
            
            if len(data_clean) == 0:
                print(f"\nWARNING: No complete cases for {method_name}")
                continue
            
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data_clean.T)
            
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(data_scaled)
            
            print(f"\n{method_name} PCA:")
            print(f"  PC1 variance: {pca.explained_variance_ratio_[0]:.2%}")
            print(f"  PC2 variance: {pca.explained_variance_ratio_[1]:.2%}")
            print(f"  Total variance: {sum(pca.explained_variance_ratio_):.2%}")
            
            ax = axes[idx]
            colors = {'LC': 'blue', 'NC': 'green', 'PTB': 'red', 'QC': 'orange'}
            
            sample_to_group = {}
            for group, samples in self.groups.items():
                for sample in samples:
                    sample_to_group[sample] = group
            
            for i, sample in enumerate(data_clean.columns):
                group = sample_to_group.get(sample, 'Unknown')
                ax.scatter(pca_result[i, 0], pca_result[i, 1], 
                          c=colors.get(group, 'gray'), label=group,
                          alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
            
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best', frameon=True, fontsize=8)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontweight='bold')
            ax.set_title(f'{method_name} Imputation', fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                self.comparison_results['knn_pca_variance'] = sum(pca.explained_variance_ratio_)
            elif idx == 1:
                self.comparison_results['em_pca_variance'] = sum(pca.explained_variance_ratio_)
            elif idx == 2:
                self.comparison_results['rf_pca_variance'] = sum(pca.explained_variance_ratio_)
            elif idx == 3:
                self.comparison_results['svd_pca_variance'] = sum(pca.explained_variance_ratio_)
            else:
                self.comparison_results['halfmin_pca_variance'] = sum(pca.explained_variance_ratio_)
        
        axes[-1].axis('off')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pca_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved: {output_dir}/pca_comparison.png")
        plt.close()
    
    def compare_variance_preservation(self, output_dir='comparison_results'):
        """Compare how well each method preserves feature-wise variance"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("VARIANCE PRESERVATION COMPARISON")
        print("="*70)
        
        original_clean = self.original_data.copy()
        for col in original_clean.columns:
            original_clean[col] = pd.to_numeric(original_clean[col], errors='coerce')
        original_clean = original_clean.replace(0, np.nan)
        
        features_with_data = original_clean.notna().sum(axis=1) > 0.5 * len(original_clean.columns)
        
        original_var = original_clean.loc[features_with_data].var(axis=1, skipna=True)
        knn_var = self.knn_data.loc[features_with_data].var(axis=1, skipna=True)
        em_var = self.em_data.loc[features_with_data].var(axis=1, skipna=True)
        rf_var = self.rf_data.loc[features_with_data].var(axis=1, skipna=True)
        svd_var = self.svd_data.loc[features_with_data].var(axis=1, skipna=True)
        halfmin_var = self.halfmin_data.loc[features_with_data].var(axis=1, skipna=True)
        
        valid_idx = (original_var.notna() & knn_var.notna() & em_var.notna() & 
                     rf_var.notna() & svd_var.notna() & halfmin_var.notna())
        original_var = original_var[valid_idx]
        knn_var = knn_var[valid_idx]
        em_var = em_var[valid_idx]
        rf_var = rf_var[valid_idx]
        svd_var = svd_var[valid_idx]
        halfmin_var = halfmin_var[valid_idx]
        
        knn_corr = np.corrcoef(original_var, knn_var)[0, 1]
        em_corr = np.corrcoef(original_var, em_var)[0, 1]
        rf_corr = np.corrcoef(original_var, rf_var)[0, 1]
        svd_corr = np.corrcoef(original_var, svd_var)[0, 1]
        halfmin_corr = np.corrcoef(original_var, halfmin_var)[0, 1]
        
        print(f"\nVariance preservation (correlation with original):")
        print(f"  KNN:     {knn_corr:.4f}")
        print(f"  EM:      {em_corr:.4f}")
        print(f"  RF:      {rf_corr:.4f}")
        print(f"  SVD:     {svd_corr:.4f}")
        print(f"  HalfMin: {halfmin_corr:.4f}")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        methods = [('KNN', knn_var, knn_corr, 'blue'),
                   ('EM', em_var, em_corr, 'green'),
                   ('RF', rf_var, rf_corr, 'orange'),
                   ('SVD', svd_var, svd_corr, 'purple'),
                   ('Half-Min', halfmin_var, halfmin_corr, 'red')]
        
        for idx, (name, var_data, corr, color) in enumerate(methods):
            ax = axes[idx]
            ax.scatter(np.log10(original_var + 1), np.log10(var_data + 1), 
                      alpha=0.5, s=20, color=color, label=f'{name} (r={corr:.3f})')
            lims = [ax.get_xlim(), ax.get_ylim()]
            lims = [min(lims[0][0], lims[1][0]), max(lims[0][1], lims[1][1])]
            ax.plot(lims, lims, 'r--', alpha=0.7, linewidth=2)
            ax.set_xlabel('log10(Original Variance + 1)', fontweight='bold')
            ax.set_ylabel(f'log10({name} Variance + 1)', fontweight='bold')
            ax.set_title(f'{name} Variance Preservation', fontweight='bold', pad=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/variance_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved: {output_dir}/variance_comparison.png")
        plt.close()
        
        self.comparison_results['variance'] = {
            'knn_correlation': knn_corr,
            'em_correlation': em_corr,
            'rf_correlation': rf_corr,
            'svd_correlation': svd_corr,
            'halfmin_correlation': halfmin_corr
        }
    
    def compare_feature_correlations(self, output_dir='comparison_results', n_features=50):
        """Compare correlation structure preservation"""
        Path(output_dir).mkdir(exist_ok=True)
        
        print("\n" + "="*70)
        print("CORRELATION STRUCTURE COMPARISON")
        print("="*70)
        
        original_clean = self.original_data.copy()
        for col in original_clean.columns:
            original_clean[col] = pd.to_numeric(original_clean[col], errors='coerce')
        original_clean = original_clean.replace(0, np.nan)
        
        features_with_data = original_clean.notna().sum(axis=1) >= 0.8 * len(original_clean.columns)
        selected_features = original_clean.loc[features_with_data].head(n_features)
        
        original_corr = selected_features.T.corr()
        knn_corr = self.knn_data.loc[selected_features.index].T.corr()
        em_corr = self.em_data.loc[selected_features.index].T.corr()
        rf_corr = self.rf_data.loc[selected_features.index].T.corr()
        svd_corr = self.svd_data.loc[selected_features.index].T.corr()
        halfmin_corr = self.halfmin_data.loc[selected_features.index].T.corr()
        
        mask = np.triu(np.ones_like(original_corr, dtype=bool), k=1)
        original_corr_vals = original_corr.values[mask]
        knn_corr_vals = knn_corr.values[mask]
        em_corr_vals = em_corr.values[mask]
        rf_corr_vals = rf_corr.values[mask]
        svd_corr_vals = svd_corr.values[mask]
        halfmin_corr_vals = halfmin_corr.values[mask]
        
        valid_idx = (~np.isnan(original_corr_vals) & ~np.isnan(knn_corr_vals) & 
                     ~np.isnan(em_corr_vals) & ~np.isnan(rf_corr_vals) & 
                     ~np.isnan(svd_corr_vals) & ~np.isnan(halfmin_corr_vals))
        original_corr_vals = original_corr_vals[valid_idx]
        knn_corr_vals = knn_corr_vals[valid_idx]
        em_corr_vals = em_corr_vals[valid_idx]
        rf_corr_vals = rf_corr_vals[valid_idx]
        svd_corr_vals = svd_corr_vals[valid_idx]
        halfmin_corr_vals = halfmin_corr_vals[valid_idx]
        
        knn_corr_preservation = np.corrcoef(original_corr_vals, knn_corr_vals)[0, 1]
        em_corr_preservation = np.corrcoef(original_corr_vals, em_corr_vals)[0, 1]
        rf_corr_preservation = np.corrcoef(original_corr_vals, rf_corr_vals)[0, 1]
        svd_corr_preservation = np.corrcoef(original_corr_vals, svd_corr_vals)[0, 1]
        halfmin_corr_preservation = np.corrcoef(original_corr_vals, halfmin_corr_vals)[0, 1]
        
        print(f"\nCorrelation structure preservation:")
        print(f"  KNN:     {knn_corr_preservation:.4f}")
        print(f"  EM:      {em_corr_preservation:.4f}")
        print(f"  RF:      {rf_corr_preservation:.4f}")
        print(f"  SVD:     {svd_corr_preservation:.4f}")
        print(f"  HalfMin: {halfmin_corr_preservation:.4f}")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        methods = [('KNN', knn_corr_vals, knn_corr_preservation, 'blue'),
                   ('EM', em_corr_vals, em_corr_preservation, 'green'),
                   ('RF', rf_corr_vals, rf_corr_preservation, 'orange'),
                   ('SVD', svd_corr_vals, svd_corr_preservation, 'purple'),
                   ('Half-Min', halfmin_corr_vals, halfmin_corr_preservation, 'red')]
        
        for idx, (name, corr_vals, preservation, color) in enumerate(methods):
            ax = axes[idx]
            ax.scatter(original_corr_vals, corr_vals, alpha=0.3, s=10, color=color)
            ax.plot([-1, 1], [-1, 1], 'r--', alpha=0.7, linewidth=2)
            ax.set_xlabel('Original Correlation', fontweight='bold')
            ax.set_ylabel(f'{name} Correlation', fontweight='bold')
            ax.set_title(f'{name} (r={preservation:.3f})', fontweight='bold', pad=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/correlation_comparison.png', dpi=300, bbox_inches='tight')
        print(f"\n[OK] Saved: {output_dir}/correlation_comparison.png")
        plt.close()
        
        self.comparison_results['correlation_structure'] = {
            'knn_preservation': knn_corr_preservation,
            'em_preservation': em_corr_preservation,
            'rf_preservation': rf_corr_preservation,
            'svd_preservation': svd_corr_preservation,
            'halfmin_preservation': halfmin_corr_preservation
        }
    
    def generate_final_report(self, output_dir='comparison_results'):
        """Generate comprehensive comparison report"""
        Path(output_dir).mkdir(exist_ok=True)
        
        report = []
        report.append("="*70)
        report.append("IMPUTATION METHODS COMPARISON: KNN vs EM vs RF vs SVD vs Half-Min")
        report.append("="*70)
        
        report.append("\n" + "="*70)
        report.append("1. DISTRIBUTION ANALYSIS")
        report.append("="*70)
        
        if 'distribution' in self.comparison_results:
            dist = self.comparison_results['distribution']
            report.append(f"\nImputed values statistics:")
            report.append(f"  KNN:     mean = {dist['knn_mean']:.2f}, std = {dist['knn_std']:.2f}")
            report.append(f"  EM:      mean = {dist['em_mean']:.2f}, std = {dist['em_std']:.2f}")
            report.append(f"  RF:      mean = {dist['rf_mean']:.2f}, std = {dist['rf_std']:.2f}")
            report.append(f"  SVD:     mean = {dist['svd_mean']:.2f}, std = {dist['svd_std']:.2f}")
            report.append(f"  HalfMin: mean = {dist['halfmin_mean']:.2f}, std = {dist['halfmin_std']:.2f}")
        
        report.append("\n" + "="*70)
        report.append("2. VARIANCE PRESERVATION")
        report.append("="*70)
        
        if 'variance' in self.comparison_results:
            var = self.comparison_results['variance']
            report.append(f"\nCorrelation with original feature variances:")
            report.append(f"  KNN:     {var['knn_correlation']:.4f}")
            report.append(f"  EM:      {var['em_correlation']:.4f}")
            report.append(f"  RF:      {var['rf_correlation']:.4f}")
            report.append(f"  SVD:     {var['svd_correlation']:.4f}")
            report.append(f"  HalfMin: {var['halfmin_correlation']:.4f}")
            
            best_var = max(var['knn_correlation'], var['em_correlation'], var['rf_correlation'], 
                          var['svd_correlation'], var['halfmin_correlation'])
            if var['knn_correlation'] == best_var:
                report.append(f"\n  Winner: KNN")
            elif var['em_correlation'] == best_var:
                report.append(f"\n  Winner: EM")
            elif var['rf_correlation'] == best_var:
                report.append(f"\n  Winner: RF")
            elif var['svd_correlation'] == best_var:
                report.append(f"\n  Winner: SVD")
            else:
                report.append(f"\n  Winner: Half-Min")
        
        report.append("\n" + "="*70)
        report.append("3. CORRELATION STRUCTURE")
        report.append("="*70)
        
        if 'correlation_structure' in self.comparison_results:
            corr = self.comparison_results['correlation_structure']
            report.append(f"\nPreservation of pairwise feature correlations:")
            report.append(f"  KNN:     {corr['knn_preservation']:.4f}")
            report.append(f"  EM:      {corr['em_preservation']:.4f}")
            report.append(f"  RF:      {corr['rf_preservation']:.4f}")
            report.append(f"  SVD:     {corr['svd_preservation']:.4f}")
            report.append(f"  HalfMin: {corr['halfmin_preservation']:.4f}")
            
            best_corr = max(corr['knn_preservation'], corr['em_preservation'], corr['rf_preservation'], 
                           corr['svd_preservation'], corr['halfmin_preservation'])
            if corr['knn_preservation'] == best_corr:
                report.append(f"\n  Winner: KNN")
            elif corr['em_preservation'] == best_corr:
                report.append(f"\n  Winner: EM")
            elif corr['rf_preservation'] == best_corr:
                report.append(f"\n  Winner: RF")
            elif corr['svd_preservation'] == best_corr:
                report.append(f"\n  Winner: SVD")
            else:
                report.append(f"\n  Winner: Half-Min")
        
        report.append("\n" + "="*70)
        report.append("4. GROUP SEPARATION (PCA)")
        report.append("="*70)
        
        if all(k in self.comparison_results for k in ['knn_pca_variance', 'em_pca_variance', 'rf_pca_variance', 'svd_pca_variance', 'halfmin_pca_variance']):
            knn_pca = self.comparison_results['knn_pca_variance']
            em_pca = self.comparison_results['em_pca_variance']
            rf_pca = self.comparison_results['rf_pca_variance']
            svd_pca = self.comparison_results['svd_pca_variance']
            halfmin_pca = self.comparison_results['halfmin_pca_variance']
            report.append(f"\nVariance explained by first 2 PCs:")
            report.append(f"  KNN:     {knn_pca:.2%}")
            report.append(f"  EM:      {em_pca:.2%}")
            report.append(f"  RF:      {rf_pca:.2%}")
            report.append(f"  SVD:     {svd_pca:.2%}")
            report.append(f"  HalfMin: {halfmin_pca:.2%}")
            
            best_pca = max(knn_pca, em_pca, rf_pca, svd_pca, halfmin_pca)
            if knn_pca == best_pca:
                report.append(f"\n  Winner: KNN")
            elif em_pca == best_pca:
                report.append(f"\n  Winner: EM")
            elif rf_pca == best_pca:
                report.append(f"\n  Winner: RF")
            elif svd_pca == best_pca:
                report.append(f"\n  Winner: SVD")
            else:
                report.append(f"\n  Winner: Half-Min")
        
        report.append("\n" + "="*70)
        report.append("FINAL RECOMMENDATION")
        report.append("="*70)
        
        scores = {'KNN': 0, 'EM': 0, 'RF': 0, 'SVD': 0, 'HalfMin': 0}
        
        if 'variance' in self.comparison_results:
            var = self.comparison_results['variance']
            best_var = max(var['knn_correlation'], var['em_correlation'], var['rf_correlation'], 
                          var['svd_correlation'], var['halfmin_correlation'])
            if var['knn_correlation'] == best_var:
                scores['KNN'] += 1
            elif var['em_correlation'] == best_var:
                scores['EM'] += 1
            elif var['rf_correlation'] == best_var:
                scores['RF'] += 1
            elif var['svd_correlation'] == best_var:
                scores['SVD'] += 1
            else:
                scores['HalfMin'] += 1
        
        if 'correlation_structure' in self.comparison_results:
            corr = self.comparison_results['correlation_structure']
            best_corr = max(corr['knn_preservation'], corr['em_preservation'], corr['rf_preservation'], 
                           corr['svd_preservation'], corr['halfmin_preservation'])
            if corr['knn_preservation'] == best_corr:
                scores['KNN'] += 1
            elif corr['em_preservation'] == best_corr:
                scores['EM'] += 1
            elif corr['rf_preservation'] == best_corr:
                scores['RF'] += 1
            elif corr['svd_preservation'] == best_corr:
                scores['SVD'] += 1
            else:
                scores['HalfMin'] += 1
        
        if all(k in self.comparison_results for k in ['knn_pca_variance', 'em_pca_variance', 'rf_pca_variance', 'svd_pca_variance', 'halfmin_pca_variance']):
            knn_pca = self.comparison_results['knn_pca_variance']
            em_pca = self.comparison_results['em_pca_variance']
            rf_pca = self.comparison_results['rf_pca_variance']
            svd_pca = self.comparison_results['svd_pca_variance']
            halfmin_pca = self.comparison_results['halfmin_pca_variance']
            best_pca = max(knn_pca, em_pca, rf_pca, svd_pca, halfmin_pca)
            if knn_pca == best_pca:
                scores['KNN'] += 1
            elif em_pca == best_pca:
                scores['EM'] += 1
            elif rf_pca == best_pca:
                scores['RF'] += 1
            elif svd_pca == best_pca:
                scores['SVD'] += 1
            else:
                scores['HalfMin'] += 1
        
        report.append(f"\nScore: KNN={scores['KNN']}, EM={scores['EM']}, RF={scores['RF']}, SVD={scores['SVD']}, HalfMin={scores['HalfMin']}")
        
        winner = max(scores, key=scores.get)
        report.append(f"\nRecommendation: {winner} imputation")
        if winner == 'KNN':
            report.append("  - Best variance preservation")
            report.append("  - Superior correlation structure")
            report.append("  - Non-parametric and robust")
        elif winner == 'EM':
            report.append("  - Best statistical foundation")
            report.append("  - Strong theoretical guarantees")
            report.append("  - Optimal under Gaussian assumptions")
        elif winner == 'RF':
            report.append("  - Best non-linear relationship capture")
            report.append("  - Robust to distribution assumptions")
            report.append("  - Handles complex interactions")
        elif winner == 'SVD':
            report.append("  - Best low-rank structure capture")
            report.append("  - Efficient dimensionality reduction")
            report.append("  - Good for highly correlated features")
        else:
            report.append("  - Simple and fast baseline")
            report.append("  - Domain-specific for metabolomics")
            report.append("  - Conservative for low-abundance values")
            report.append("  - Note: Sophisticated methods typically outperform")
        
        report.append("\n" + "="*70)
        
        report_text = "\n".join(report)
        print("\n" + report_text)
        
        with open(f'{output_dir}/comparison_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)
        
        print(f"\n[OK] Saved: {output_dir}/comparison_report.txt")


def main():
    """Main comparison workflow"""
    print("="*70)
    print("IMPUTATION METHODS COMPARISON: KNN vs EM vs RF vs SVD vs Half-Min")
    print("="*70)
    
    csv_file = '../../data/progenesis_file.csv'
    knn_file = 'knn_imputation_results/imputed_data_knn.csv'
    em_file = 'em_imputation_results/imputed_data_em.csv'
    rf_file = 'rf_imputation_results/imputed_data_rf.csv'
    svd_file = 'svd_imputation_results/imputed_data_svd.csv'
    halfmin_file = 'halfmin_imputation_results/imputed_data_halfmin.csv'
    
    print("\nLoading original data...")
    loader = ProgenesisDataLoader(csv_file)
    metadata, original_abundance, groups = loader.load_data()
    
    print("Loading KNN imputed data...")
    knn_full = pd.read_csv(knn_file)
    
    metadata_cols = metadata.columns.tolist()
    abundance_cols = [col for col in knn_full.columns if col not in metadata_cols]
    knn_abundance = knn_full[abundance_cols]
    
    knn_abundance.columns = original_abundance.columns
    
    print("Loading EM imputed data...")
    em_full = pd.read_csv(em_file)
    em_abundance = em_full[abundance_cols]
    
    em_abundance.columns = original_abundance.columns
    
    print("Loading RF imputed data...")
    rf_full = pd.read_csv(rf_file)
    rf_abundance = rf_full[abundance_cols]
    
    rf_abundance.columns = original_abundance.columns
    
    print("Loading SVD imputed data...")
    svd_full = pd.read_csv(svd_file)
    svd_abundance = svd_full[abundance_cols]
    
    svd_abundance.columns = original_abundance.columns
    
    print("Loading Half-Min imputed data...")
    halfmin_full = pd.read_csv(halfmin_file)
    halfmin_abundance = halfmin_full[abundance_cols]
    
    halfmin_abundance.columns = original_abundance.columns
    
    print("\nInitializing comparison framework...")
    comparator = ImputationComparator(
        original_abundance, 
        knn_abundance, 
        em_abundance,
        rf_abundance,
        svd_abundance,
        halfmin_abundance, 
        groups, 
        metadata
    )
    
    print("\nStep 1: Comparing imputed value distributions")
    comparator.compare_imputed_distributions()
    
    print("\nStep 2: Comparing group separation (PCA)")
    comparator.compare_group_separation()
    
    print("\nStep 3: Comparing variance preservation")
    comparator.compare_variance_preservation()
    
    print("\nStep 4: Comparing correlation structure")
    comparator.compare_feature_correlations()
    
    print("\nStep 5: Generating final comparison report")
    comparator.generate_final_report()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()