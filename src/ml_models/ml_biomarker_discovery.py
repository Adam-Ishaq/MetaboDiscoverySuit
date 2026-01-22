"""
MetaboAI - Machine Learning Biomarker Discovery Module
Uses AI/ML to identify optimal biomarker panels for disease classification.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, roc_curve, 
                            confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Install with: pip install shap")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLBiomarkerDiscovery:
    """
    Machine Learning-based biomarker discovery.
    
    Capabilities:
    - Train multiple ML models
    - Feature importance ranking
    - Cross-validation
    - SHAP analysis
    - Biomarker panel selection
    """
    
    def __init__(self, random_state: int = 42):
        """Initialize the ML discovery module."""
        self.random_state = random_state
        self.data = None
        self.labels = None
        self.feature_names = None
        self.models = {}
        self.results = {}
        
        logger.info(f"MLBiomarkerDiscovery initialized (random_state={random_state})")
    
    def load_data(self,
                  feature_table_path: str,
                  sample_metadata_path: str,
                  group_column: str = 'group',
                  positive_class: str = 'Semi',
                  negative_class: str = 'Naive'):
        """
        Load preprocessed data for ML analysis.
        
        Args:
            feature_table_path (str): Path to feature table
            sample_metadata_path (str): Path to sample metadata
            group_column (str): Column name for groups
            positive_class (str): Positive class label (disease)
            negative_class (str): Negative class label (control)
        """
        logger.info("Loading data for ML analysis...")
        
        # Load feature table
        self.data = pd.read_csv(feature_table_path, index_col=0)
        
        # Load metadata
        metadata = pd.read_csv(sample_metadata_path)
        if 'sample_id' in metadata.columns:
            metadata = metadata.set_index('sample_id')
        
        # Filter for only the two classes
        class_samples = metadata[
            metadata[group_column].isin([positive_class, negative_class])
        ].index
        
        # Only keep samples that exist in both metadata and data
        common_samples = class_samples.intersection(self.data.index)
        
        if len(common_samples) < len(class_samples):
            missing = set(class_samples) - set(common_samples)
            logger.warning(f"  Missing samples in data: {missing}")
        
        self.data = self.data.loc[common_samples]
        metadata = metadata.loc[common_samples]
        
        # Create binary labels
        self.labels = (metadata[group_column] == positive_class).astype(int).values
        self.feature_names = self.data.columns.tolist()
        
        logger.info(f"  Loaded {len(self.data)} samples × {len(self.feature_names)} features")
        logger.info(f"  Positive class ({positive_class}): {sum(self.labels)} samples")
        logger.info(f"  Negative class ({negative_class}): {len(self.labels) - sum(self.labels)} samples")
    
    def select_top_features(self,
                          differential_results_path: str,
                          n_features: int = 100,
                          criteria: str = 'combined') -> List[str]:
        """
        Select top features based on statistical results.
        
        Args:
            differential_results_path (str): Path to differential analysis results
            n_features (int): Number of features to select
            criteria (str): Selection criteria ('pvalue', 'foldchange', 'combined')
            
        Returns:
            list: Selected feature IDs
        """
        logger.info(f"Selecting top {n_features} features (criteria: {criteria})...")
        
        diff_results = pd.read_csv(differential_results_path)
        
        if criteria == 'pvalue':
            top_features = diff_results.nsmallest(n_features, 'p_value_adjusted')
        elif criteria == 'foldchange':
            diff_results['abs_log2fc'] = np.abs(diff_results['log2_fold_change'])
            top_features = diff_results.nlargest(n_features, 'abs_log2fc')
        elif criteria == 'combined':
            diff_results['combined_score'] = (
                -np.log10(diff_results['p_value_adjusted'] + 1e-300) * 
                np.abs(diff_results['log2_fold_change'])
            )
            top_features = diff_results.nlargest(n_features, 'combined_score')
        
        selected_features = top_features['feature_id'].tolist()
        
        # Filter data to selected features
        self.data = self.data[selected_features]
        self.feature_names = selected_features
        
        logger.info(f"✓ Selected {len(selected_features)} features")
        
        return selected_features
    
    def train_random_forest(self,
                          n_estimators: int = 500,
                          max_depth: int = 10,
                          cv_folds: int = 5) -> Dict:
        """
        Train Random Forest classifier with cross-validation.
        
        Args:
            n_estimators (int): Number of trees
            max_depth (int): Maximum tree depth
            cv_folds (int): Number of CV folds
            
        Returns:
            dict: Training results
        """
        logger.info("Training Random Forest classifier...")
        
        # Initialize model
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=-1
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = cross_val_score(rf, self.data, self.labels, cv=cv, scoring='roc_auc')
        cv_predictions = cross_val_predict(rf, self.data, self.labels, cv=cv)
        
        # Train on full data
        rf.fit(self.data, self.labels)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results
        results = {
            'model': rf,
            'cv_scores': cv_scores,
            'cv_predictions': cv_predictions,
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std(),
            'feature_importance': feature_importance
        }
        
        self.models['random_forest'] = results
        
        logger.info(f"✓ Random Forest trained: AUC = {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
        
        return results
    
    def train_xgboost(self,
                     n_estimators: int = 300,
                     max_depth: int = 6,
                     cv_folds: int = 5) -> Dict:
        """
        Train XGBoost classifier with cross-validation.
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth
            cv_folds (int): Number of CV folds
            
        Returns:
            dict: Training results
        """
        if not XGBOOST_AVAILABLE:
            logger.warning("XGBoost not available. Skipping.")
            return {}
        
        logger.info("Training XGBoost classifier...")
        
        # Initialize model
        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=0.1,
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = cross_val_score(xgb_model, self.data, self.labels, cv=cv, scoring='roc_auc')
        cv_predictions = cross_val_predict(xgb_model, self.data, self.labels, cv=cv)
        
        # Train on full data
        xgb_model.fit(self.data, self.labels)
        
        # Get feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results
        results = {
            'model': xgb_model,
            'cv_scores': cv_scores,
            'cv_predictions': cv_predictions,
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std(),
            'feature_importance': feature_importance
        }
        
        self.models['xgboost'] = results
        
        logger.info(f"✓ XGBoost trained: AUC = {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
        
        return results
    
    def train_logistic_regression(self, cv_folds: int = 5) -> Dict:
        """
        Train Logistic Regression classifier (baseline).
        
        Args:
            cv_folds (int): Number of CV folds
            
        Returns:
            dict: Training results
        """
        logger.info("Training Logistic Regression classifier...")
        
        # Initialize model
        lr = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced',
            max_iter=1000
        )
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_scores = cross_val_score(lr, self.data, self.labels, cv=cv, scoring='roc_auc')
        cv_predictions = cross_val_predict(lr, self.data, self.labels, cv=cv)
        
        # Train on full data
        lr.fit(self.data, self.labels)
        
        # Get feature importance (coefficients)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': np.abs(lr.coef_[0])
        }).sort_values('importance', ascending=False)
        
        # Store results
        results = {
            'model': lr,
            'cv_scores': cv_scores,
            'cv_predictions': cv_predictions,
            'mean_auc': cv_scores.mean(),
            'std_auc': cv_scores.std(),
            'feature_importance': feature_importance
        }
        
        self.models['logistic_regression'] = results
        
        logger.info(f"✓ Logistic Regression trained: AUC = {results['mean_auc']:.3f} ± {results['std_auc']:.3f}")
        
        return results
    
    def calculate_shap_values(self, model_name: str = 'random_forest') -> Optional[np.ndarray]:
        """
        Calculate SHAP values for model interpretability.
        
        Args:
            model_name (str): Which model to explain
            
        Returns:
            np.ndarray: SHAP values
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available. Skipping.")
            return None
        
        if model_name not in self.models:
            logger.error(f"Model {model_name} not trained")
            return None
        
        logger.info(f"Calculating SHAP values for {model_name}...")
        
        model = self.models[model_name]['model']
        
        # Create explainer
        if model_name == 'random_forest':
            explainer = shap.TreeExplainer(model)
        elif model_name == 'xgboost':
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.LinearExplainer(model, self.data)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(self.data)
        
        # For binary classification, take positive class
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        
        # Store results
        self.models[model_name]['shap_values'] = shap_values
        self.models[model_name]['shap_explainer'] = explainer
        
        logger.info("✓ SHAP values calculated")
        
        return shap_values
    
    def get_biomarker_panel(self,
                          model_name: str = 'random_forest',
                          n_biomarkers: int = 10) -> pd.DataFrame:
        """
        Select optimal biomarker panel.
        
        Args:
            model_name (str): Which model to use
            n_biomarkers (int): Number of biomarkers
            
        Returns:
            DataFrame: Biomarker panel
        """
        if model_name not in self.models:
            logger.error(f"Model {model_name} not trained")
            return pd.DataFrame()
        
        logger.info(f"Selecting top {n_biomarkers} biomarkers from {model_name}...")
        
        # Get feature importance
        importance_df = self.models[model_name]['feature_importance']
        
        # Select top N
        biomarker_panel = importance_df.head(n_biomarkers).copy()
        
        logger.info("✓ Biomarker panel selected")
        
        return biomarker_panel
    
    def evaluate_models(self) -> pd.DataFrame:
        """
        Compare all trained models.
        
        Returns:
            DataFrame: Model comparison
        """
        logger.info("Evaluating all models...")
        
        results_list = []
        
        for model_name, model_results in self.models.items():
            cv_pred = model_results['cv_predictions']
            
            metrics = {
                'model': model_name,
                'mean_auc': model_results['mean_auc'],
                'std_auc': model_results['std_auc'],
                'accuracy': accuracy_score(self.labels, cv_pred),
                'precision': precision_score(self.labels, cv_pred),
                'recall': recall_score(self.labels, cv_pred),
                'f1_score': f1_score(self.labels, cv_pred)
            }
            
            results_list.append(metrics)
        
        comparison_df = pd.DataFrame(results_list).sort_values('mean_auc', ascending=False)
        
        logger.info("Model comparison:")
        for _, row in comparison_df.iterrows():
            logger.info(f"  {row['model']:20s}: AUC={row['mean_auc']:.3f}, F1={row['f1_score']:.3f}")
        
        return comparison_df
    
    def export_results(self, output_dir: str):
        """
        Export all ML results.
        
        Args:
            output_dir (str): Output directory
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export model comparison
        comparison = self.evaluate_models()
        comparison.to_csv(output_dir / "model_comparison.csv", index=False)
        logger.info(f"✓ Model comparison saved")
        
        # Export feature importance for each model
        for model_name, results in self.models.items():
            importance_path = output_dir / f"{model_name}_feature_importance.csv"
            results['feature_importance'].to_csv(importance_path, index=False)
            logger.info(f"✓ {model_name} importance saved")
        
        # Export biomarker panels
        for model_name in self.models.keys():
            panel = self.get_biomarker_panel(model_name, n_biomarkers=20)
            panel_path = output_dir / f"{model_name}_biomarker_panel.csv"
            panel.to_csv(panel_path, index=False)
            logger.info(f"✓ {model_name} biomarker panel saved")
        
        logger.info(f"All results saved to: {output_dir}")


# Example usage
if __name__ == "__main__":
    print("ML Biomarker Discovery Module")
    print("=" * 70)
    print()
    print("This module uses machine learning for biomarker discovery.")
    print()
    print("=" * 70)