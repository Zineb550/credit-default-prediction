"""
Model Evaluation Module
Advanced model evaluation, analysis, and interpretation tools.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    roc_auc_score, f1_score, matthews_corrcoef
)
from sklearn.calibration import calibration_curve
import logging
from typing import Dict, List, Tuple, Any
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Advanced model evaluation and interpretation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize evaluator."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def evaluate_model_comprehensive(self, model: Any, X_test: pd.DataFrame, 
                                    y_test: pd.Series, model_name: str) -> Dict:
        """Comprehensive model evaluation."""
        logger.info(f"Performing comprehensive evaluation for {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
        
        # Basic metrics
        metrics = {
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba),
            'f1_score': f1_score(y_test, y_pred),
            'mcc': matthews_corrcoef(y_test, y_pred)
        }
        
        # Cost-sensitive evaluation
        if self.config['evaluation'].get('cost_sensitive_evaluation', True):
            metrics['business_metrics'] = self.calculate_business_metrics(
                y_test, y_pred, y_proba
            )
        
        # Threshold analysis
        if self.config['evaluation'].get('threshold_optimization', True):
            metrics['optimal_threshold'] = self.find_optimal_threshold(y_test, y_proba)
        
        return metrics
    
    def calculate_business_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  y_proba: np.ndarray) -> Dict:
        """Calculate business-relevant metrics considering costs."""
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Cost configuration
        fp_cost = self.config['evaluation'].get('false_positive_cost', 1)
        fn_cost = self.config['evaluation'].get('false_negative_cost', 5)
        
        total_cost = (fp * fp_cost) + (fn * fn_cost)
        
        # Calculate profit/loss metrics
        total_samples = len(y_true)
        default_rate = y_true.sum() / total_samples
        
        metrics = {
            'total_cost': total_cost,
            'false_positive_cost': fp * fp_cost,
            'false_negative_cost': fn * fn_cost,
            'cost_per_sample': total_cost / total_samples,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'default_rate': default_rate,
            'predicted_default_rate': y_pred.sum() / total_samples
        }
        
        return metrics
    
    def find_optimal_threshold(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict:
        """Find optimal classification threshold."""
        # Calculate metrics for different thresholds
        thresholds = np.arange(0.1, 0.9, 0.05)
        results = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            # Cost-sensitive metric
            fp_cost = self.config['evaluation'].get('false_positive_cost', 1)
            fn_cost = self.config['evaluation'].get('false_negative_cost', 5)
            total_cost = (fp * fp_cost) + (fn * fn_cost)
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'total_cost': total_cost,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            })
        
        # Find threshold with minimum cost
        optimal_idx = min(range(len(results)), key=lambda i: results[i]['total_cost'])
        optimal = results[optimal_idx]
        
        logger.info(f"Optimal threshold: {optimal['threshold']:.2f}")
        logger.info(f"At optimal threshold - Precision: {optimal['precision']:.4f}, "
                   f"Recall: {optimal['recall']:.4f}, F1: {optimal['f1']:.4f}")
        
        return optimal
    
    def plot_calibration_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                              model_name: str, output_path: str):
        """Plot calibration curve to assess probability calibration."""
        prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=10)
        
        plt.figure(figsize=(10, 7))
        plt.plot(prob_pred, prob_true, marker='o', linewidth=2, label=model_name)
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')
        
        plt.xlabel('Mean Predicted Probability', fontsize=12)
        plt.ylabel('Fraction of Positives', fontsize=12)
        plt.title(f'Calibration Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Calibration curve saved to {output_path}")
    
    def plot_threshold_analysis(self, y_true: np.ndarray, y_proba: np.ndarray,
                               output_path: str):
        """Plot threshold analysis showing trade-offs."""
        thresholds = np.arange(0.0, 1.0, 0.01)
        precisions, recalls, f1_scores = [], [], []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)
        
        plt.figure(figsize=(12, 7))
        plt.plot(thresholds, precisions, label='Precision', linewidth=2)
        plt.plot(thresholds, recalls, label='Recall', linewidth=2)
        plt.plot(thresholds, f1_scores, label='F1 Score', linewidth=2)
        
        plt.xlabel('Threshold', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.title('Threshold Analysis: Precision-Recall Trade-off', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Threshold analysis plot saved to {output_path}")
    
    def analyze_error_patterns(self, X_test: pd.DataFrame, y_test: pd.Series,
                              y_pred: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
        """Analyze patterns in model errors."""
        # Create error analysis dataframe
        error_df = X_test.copy()
        error_df['true_label'] = y_test.values
        error_df['predicted_label'] = y_pred
        error_df['predicted_proba'] = y_proba
        error_df['error'] = (y_test.values != y_pred).astype(int)
        error_df['error_type'] = 'Correct'
        
        # Classify error types
        error_df.loc[(y_test.values == 1) & (y_pred == 0), 'error_type'] = 'False Negative'
        error_df.loc[(y_test.values == 0) & (y_pred == 1), 'error_type'] = 'False Positive'
        
        # Analyze error patterns
        logger.info("\nError Pattern Analysis:")
        logger.info(f"Total Errors: {error_df['error'].sum()}")
        logger.info(f"False Positives: {(error_df['error_type'] == 'False Positive').sum()}")
        logger.info(f"False Negatives: {(error_df['error_type'] == 'False Negative').sum()}")
        
        # Statistical comparison of errors vs correct predictions
        error_stats = error_df.groupby('error_type').mean()
        
        return error_df, error_stats
    
    def plot_error_distribution(self, error_df: pd.DataFrame, output_path: str):
        """Plot distribution of errors across feature space."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Key features for error analysis
        features = ['age', 'DebtRatio', 'MonthlyIncome', 'RevolvingUtilizationOfUnsecuredLines']
        
        for idx, feature in enumerate(features):
            ax = axes[idx // 2, idx % 2]
            
            for error_type in ['Correct', 'False Positive', 'False Negative']:
                data = error_df[error_df['error_type'] == error_type][feature]
                ax.hist(data, alpha=0.5, label=error_type, bins=30)
            
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.set_title(f'Error Distribution: {feature}', fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Error distribution plot saved to {output_path}")
    
    def generate_model_insights(self, model: Any, X_test: pd.DataFrame,
                               feature_names: List[str]) -> Dict:
        """Generate insights about model behavior."""
        insights = {}
        
        # Feature importance (if available)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            insights['top_10_features'] = feature_importance.head(10).to_dict('records')
            insights['bottom_10_features'] = feature_importance.tail(10).to_dict('records')
        
        # Coefficient analysis for linear models
        elif hasattr(model, 'coef_'):
            coefficients = model.coef_[0]
            feature_coef = pd.DataFrame({
                'feature': feature_names,
                'coefficient': coefficients,
                'abs_coefficient': np.abs(coefficients)
            }).sort_values('abs_coefficient', ascending=False)
            
            insights['top_10_features'] = feature_coef.head(10).to_dict('records')
            insights['most_negative'] = feature_coef.nsmallest(5, 'coefficient').to_dict('records')
            insights['most_positive'] = feature_coef.nlargest(5, 'coefficient').to_dict('records')
        
        return insights
    
    def create_evaluation_dashboard(self, results: Dict, output_path: str):
        """Create comprehensive evaluation dashboard."""
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # This is a placeholder for a comprehensive dashboard
        # You can expand this with specific visualizations
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Evaluation dashboard saved to {output_path}")


if __name__ == "__main__":
    logger.info("Model Evaluator module loaded successfully")
