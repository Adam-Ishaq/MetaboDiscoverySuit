"""
Run ML Biomarker Discovery on Malaria Dataset
Train multiple ML models and identify optimal biomarker panel.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.ml_models.ml_biomarker_discovery import MLBiomarkerDiscovery


def plot_roc_curves(ml_discovery, output_dir="results/figures"):
    """Create ROC curves for all models."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = {'random_forest': 'blue', 'xgboost': 'red', 'logistic_regression': 'green'}
    
    for model_name, results in ml_discovery.models.items():
        # Get predictions
        model = results['model']
        y_pred_proba = model.predict_proba(ml_discovery.data)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(ml_discovery.labels, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot
        ax.plot(fpr, tpr, color=colors.get(model_name, 'black'),
               linewidth=2, label=f'{model_name.replace("_", " ").title()} (AUC = {roc_auc:.3f})')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random (AUC = 0.500)')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    fig_path = output_path / "roc_curves.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curves saved: {fig_path}")
    
    plt.show()


def plot_feature_importance(ml_discovery, model_name='random_forest', 
                           n_features=20, output_dir="results/figures"):
    """Plot feature importance for a model."""
    
    if model_name not in ml_discovery.models:
        print(f"Model {model_name} not found")
        return
    
    importance_df = ml_discovery.models[model_name]['feature_importance'].head(n_features)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Horizontal bar plot
    y_pos = np.arange(len(importance_df))
    ax.barh(y_pos, importance_df['importance'].values, 
           color='steelblue', edgecolor='black', alpha=0.8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f[:30] for f in importance_df['feature'].values], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {n_features} Features - {model_name.replace("_", " ").title()}',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    fig_path = output_path / f"feature_importance_{model_name}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Feature importance saved: {fig_path}")
    
    plt.show()


def plot_confusion_matrices(ml_discovery, output_dir="results/figures"):
    """Plot confusion matrices for all models."""
    
    n_models = len(ml_discovery.models)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    
    if n_models == 1:
        axes = [axes]
    
    for idx, (model_name, results) in enumerate(ml_discovery.models.items()):
        ax = axes[idx]
        
        # Get confusion matrix
        cv_pred = results['cv_predictions']
        cm = confusion_matrix(ml_discovery.labels, cv_pred)
        
        # Plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   cbar=False, square=True,
                   xticklabels=['Naive', 'Semi'],
                   yticklabels=['Naive', 'Semi'])
        
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_title(model_name.replace('_', ' ').title(), 
                    fontsize=12, fontweight='bold')
    
    plt.suptitle('Confusion Matrices - Cross-Validation', 
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    fig_path = output_path / "confusion_matrices.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrices saved: {fig_path}")
    
    plt.show()


def plot_biomarker_panel_comparison(ml_discovery, n_biomarkers=10, output_dir="results/figures"):
    """Compare biomarker panels across models."""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    all_biomarkers = set()
    model_importances = {}
    
    # Collect biomarkers from all models
    for model_name, results in ml_discovery.models.items():
        top_features = results['feature_importance'].head(n_biomarkers)
        all_biomarkers.update(top_features['feature'].tolist())
        
        # Store importances
        importance_dict = dict(zip(top_features['feature'], top_features['importance']))
        model_importances[model_name] = importance_dict
    
    all_biomarkers = sorted(list(all_biomarkers))
    
    # Create matrix
    matrix = []
    for biomarker in all_biomarkers:
        row = []
        for model_name in ml_discovery.models.keys():
            row.append(model_importances[model_name].get(biomarker, 0))
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Plot heatmap
    sns.heatmap(matrix.T, 
               xticklabels=[b[:30] for b in all_biomarkers],
               yticklabels=[m.replace('_', ' ').title() for m in ml_discovery.models.keys()],
               cmap='YlOrRd', cbar_kws={'label': 'Importance'},
               ax=ax)
    
    ax.set_xlabel('Biomarker Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Model', fontsize=12, fontweight='bold')
    ax.set_title(f'Biomarker Panel Comparison (Top {n_biomarkers} from each model)',
                fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Save
    output_path = Path(output_dir)
    fig_path = output_path / "biomarker_panel_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ Biomarker panel comparison saved: {fig_path}")
    
    plt.show()


def main():
    """Main ML biomarker discovery execution."""
    
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║       ML Biomarker Discovery - Malaria Dataset                    ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()
    
    # Initialize
    print("Initializing ML Biomarker Discovery...")
    ml_discovery = MLBiomarkerDiscovery(random_state=42)
    print()
    
    # Load data
    print("=" * 70)
    print("Step 1: Loading Data")
    print("=" * 70)
    ml_discovery.load_data(
        feature_table_path="results/preprocessed/preprocessed_feature_table.csv",
        sample_metadata_path="data/metadata/malaria_metadata.csv",
        positive_class='Semi',
        negative_class='Naive'
    )
    print()
    
    # Select top features
    print("=" * 70)
    print("Step 2: Feature Selection")
    print("=" * 70)
    selected_features = ml_discovery.select_top_features(
        differential_results_path="results/statistical_analysis/differential_analysis_results.csv",
        n_features=100,
        criteria='combined'
    )
    print()
    
    # Train models
    print("=" * 70)
    print("Step 3: Training Machine Learning Models")
    print("=" * 70)
    
    print("\n3a. Random Forest...")
    rf_results = ml_discovery.train_random_forest(
        n_estimators=500,
        max_depth=10,
        cv_folds=5
    )
    
    print("\n3b. XGBoost...")
    xgb_results = ml_discovery.train_xgboost(
        n_estimators=300,
        max_depth=6,
        cv_folds=5
    )
    
    print("\n3c. Logistic Regression...")
    lr_results = ml_discovery.train_logistic_regression(cv_folds=5)
    
    print()
    
    # Model comparison
    print("=" * 70)
    print("Step 4: Model Comparison")
    print("=" * 70)
    comparison = ml_discovery.evaluate_models()
    print("\nModel Performance Summary:")
    print(comparison.to_string(index=False))
    print()
    
    # Calculate SHAP values
    print("=" * 70)
    print("Step 5: Explainable AI (SHAP Values)")
    print("=" * 70)
    
    try:
        shap_values = ml_discovery.calculate_shap_values(model_name='random_forest')
        if shap_values is not None:
            print("✓ SHAP values calculated successfully")
        else:
            print("⚠ SHAP not available - install with: pip install shap")
    except Exception as e:
        print(f"⚠ SHAP calculation skipped: {e}")
    
    print()
    
    # Select biomarker panels
    print("=" * 70)
    print("Step 6: Biomarker Panel Selection")
    print("=" * 70)
    
    for model_name in ml_discovery.models.keys():
        panel = ml_discovery.get_biomarker_panel(model_name, n_biomarkers=10)
        print(f"\n{model_name.replace('_', ' ').title()} - Top 10 Biomarkers:")
        for i, row in panel.iterrows():
            print(f"  {i+1:2d}. {row['feature'][:40]:40s} (importance: {row['importance']:.4f})")
    
    print()
    
    # Export results
    print("=" * 70)
    print("Step 7: Exporting Results")
    print("=" * 70)
    ml_discovery.export_results("results/ml_biomarker_discovery")
    print()
    
    # Create visualizations
    print("=" * 70)
    print("Step 8: Creating Visualizations")
    print("=" * 70)
    
    print("\nCreating ROC curves...")
    plot_roc_curves(ml_discovery)
    
    print("\nCreating feature importance plots...")
    for model_name in ml_discovery.models.keys():
        plot_feature_importance(ml_discovery, model_name, n_features=20)
    
    print("\nCreating confusion matrices...")
    plot_confusion_matrices(ml_discovery)
    
    print("\nCreating biomarker panel comparison...")
    plot_biomarker_panel_comparison(ml_discovery, n_biomarkers=10)
    
    print()
    print("=" * 70)
    print("✓ ML Biomarker Discovery Complete!")
    print("=" * 70)
    print()
    print("Files created:")
    print("  Results:")
    print("    - Model comparison:              results/ml_biomarker_discovery/model_comparison.csv")
    print("    - Feature importance:            results/ml_biomarker_discovery/*_feature_importance.csv")
    print("    - Biomarker panels:              results/ml_biomarker_discovery/*_biomarker_panel.csv")
    print()
    print("  Visualizations:")
    print("    - ROC curves:                    results/figures/roc_curves.png")
    print("    - Feature importance:            results/figures/feature_importance_*.png")
    print("    - Confusion matrices:            results/figures/confusion_matrices.png")
    print("    - Biomarker comparison:          results/figures/biomarker_panel_comparison.png")
    print()
    print("🎉 CONGRATULATIONS! 🎉")
    print("You have successfully completed a full metabolomics biomarker discovery pipeline!")
    print()
    print("Summary:")
    best_model = comparison.iloc[0]
    print(f"  - Best Model: {best_model['model']}")
    print(f"  - AUC: {best_model['mean_auc']:.3f}")
    print(f"  - Accuracy: {best_model['accuracy']:.3f}")
    print(f"  - F1 Score: {best_model['f1_score']:.3f}")
    print()
    print("=" * 70)


if __name__ == "__main__":
    main()