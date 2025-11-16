"""
Main Training Script
Orchestrates the complete training pipeline for credit default prediction.
"""

import os
import sys
import yaml
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Get the project root directory (parent of src)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add project root to path
sys.path.insert(0, project_root)

# Create necessary directories
os.makedirs(os.path.join(project_root, 'logs'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'models'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'results'), exist_ok=True)
os.makedirs(os.path.join(project_root, 'data', 'processed'), exist_ok=True)

# Setup logging
log_file = os.path.join(project_root, 'logs', 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from src.preprocessing import DataPreprocessor
from src.model_trainer import ModelTrainer


class CreditDefaultPipeline:
    """Complete ML pipeline for credit default prediction."""
    
    def __init__(self, config_path=None):
        """Initialize the pipeline."""
        if config_path is None:
            config_path = os.path.join(project_root, 'config', 'config.yaml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preprocessor = DataPreprocessor(config_path)
        self.trainer = ModelTrainer(config_path)
        self.experiment_name = f"experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Initialized pipeline: {self.experiment_name}")
    
    def load_data(self) -> pd.DataFrame:
        """Load training data."""
        logger.info("\n" + "="*70)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*70)
        
        data_path = os.path.join(project_root, self.config['data']['train_path'])
        df = self.preprocessor.load_data(data_path)
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame):
        """Preprocess the data."""
        logger.info("\n" + "="*70)
        logger.info("STEP 2: PREPROCESSING DATA")
        logger.info("="*70)
        
        X_train, X_val, y_train, y_val, stats = self.preprocessor.preprocess_pipeline(
            df, is_training=True
        )
        
        logger.info("Preprocessing completed successfully!")
        
        return X_train, X_val, y_train, y_val, stats
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series):
        """Train all models."""
        logger.info("\n" + "="*70)
        logger.info("STEP 3: TRAINING MODELS")
        logger.info("="*70)
        
        # Train base models
        results = self.trainer.train_all_models(X_train, y_train, X_val, y_val)
        
        # Train ensemble models
        self.trainer.train_ensemble_models(X_train, y_train, X_val, y_val)
        
        return self.trainer.results
    
    def evaluate_models(self, results: dict, X_val: pd.DataFrame, y_val: pd.Series):
        """Evaluate and visualize model performance."""
        logger.info("\n" + "="*70)
        logger.info("STEP 4: EVALUATING MODELS")
        logger.info("="*70)
        
        # Create evaluation plots
        self.plot_roc_curves(results, y_val)
        self.plot_precision_recall_curves(results, y_val)
        self.plot_confusion_matrices(results, y_val)
        self.plot_feature_importance(X_val)
        
        # Generate simple evaluation report
        report = self.generate_evaluation_report(results)
        
        logger.info("Evaluation completed!")
        
        return report
    
    def plot_roc_curves(self, results: dict, y_val: pd.Series):
        """Plot ROC curves for all models."""
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=(12, 8))
        
        for model_name, result in results.items():
            if 'y_proba_val' in result:
                fpr, tpr, _ = roc_curve(y_val, result['y_proba_val'])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})", linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(project_root, 'results', f'{self.experiment_name}_roc_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"ROC curves saved to {plot_path}")
    
    def plot_precision_recall_curves(self, results: dict, y_val: pd.Series):
        """Plot Precision-Recall curves for all models."""
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        plt.figure(figsize=(12, 8))
        
        for model_name, result in results.items():
            if 'y_proba_val' in result:
                precision, recall, _ = precision_recall_curve(y_val, result['y_proba_val'])
                pr_auc = average_precision_score(y_val, result['y_proba_val'])
                plt.plot(recall, precision, label=f"{model_name} (AP = {pr_auc:.3f})", linewidth=2)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curves - All Models', fontsize=14, fontweight='bold')
        plt.legend(loc="best", fontsize=10)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        plot_path = os.path.join(project_root, 'results', f'{self.experiment_name}_pr_curves.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Precision-Recall curves saved to {plot_path}")
    
    def plot_confusion_matrices(self, results: dict, y_val: pd.Series):
        """Plot confusion matrices for all models."""
        from sklearn.metrics import confusion_matrix
        
        n_models = len(results)
        n_cols = 3
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_models > 1 else [axes]
        
        for idx, (model_name, result) in enumerate(results.items()):
            if idx >= len(axes):
                break
            
            cm = confusion_matrix(y_val, result['y_pred_val'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar=False, square=True)
            axes[idx].set_title(f"{model_name}", fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        # Hide extra subplots
        for idx in range(len(results), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        plot_path = os.path.join(project_root, 'results', f'{self.experiment_name}_confusion_matrices.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Confusion matrices saved to {plot_path}")
    
    def plot_feature_importance(self, X_val: pd.DataFrame):
        """Plot feature importance for tree-based models."""
        best_model_name, best_model = self.trainer.get_best_model()
        
        # Check if model has feature_importances_
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            indices = np.argsort(importances)[::-1][:20]  # Top 20 features
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Top 20 Feature Importances - {best_model_name}', 
                     fontsize=14, fontweight='bold')
            plt.bar(range(len(indices)), importances[indices])
            plt.xticks(range(len(indices)), 
                      [X_val.columns[i] for i in indices], 
                      rotation=45, ha='right')
            plt.xlabel('Features', fontsize=12)
            plt.ylabel('Importance', fontsize=12)
            plt.tight_layout()
            
            plot_path = os.path.join(project_root, 'results', f'{self.experiment_name}_feature_importance.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance plot saved to {plot_path}")
    
    def generate_evaluation_report(self, results: dict) -> dict:
        """Generate comprehensive evaluation report."""
        report = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),
            'models': {}
        }
        
        for model_name, result in results.items():
            report['models'][model_name] = {
                'train_metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                 for k, v in result['train_metrics'].items()},
                'val_metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                               for k, v in result['val_metrics'].items()},
                'training_time': float(result['training_time'])
            }
        
        # Identify best model
        best_model_name = max(results.keys(), 
                             key=lambda x: results[x]['val_metrics']['roc_auc'])
        
        report['best_model'] = {
            'name': best_model_name,
            'metrics': {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                       for k, v in results[best_model_name]['val_metrics'].items()}
        }
        
        return report
    
    def save_models(self):
        """Save all trained models."""
        logger.info("\n" + "="*70)
        logger.info("STEP 5: SAVING MODELS")
        logger.info("="*70)
        
        # Save all models
        models_dir = os.path.join(project_root, 'models')
        self.trainer.save_all_models(output_dir=models_dir)
        
        # Save preprocessor
        preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
        self.preprocessor.save_preprocessor(preprocessor_path)
        
        logger.info("All models and artifacts saved successfully!")
    
    def run_complete_pipeline(self):
        """Run the complete ML pipeline."""
        logger.info("\n" + "="*70)
        logger.info("STARTING CREDIT DEFAULT PREDICTION PIPELINE")
        logger.info("="*70)
        logger.info(f"Experiment: {self.experiment_name}")
        
        start_time = datetime.now()
        
        try:
            # Load data
            df = self.load_data()
            
            # Preprocess data
            X_train, X_val, y_train, y_val, stats = self.preprocess_data(df)
            
            # Train models
            results = self.train_models(X_train, y_train, X_val, y_val)
            
            # Evaluate models
            report = self.evaluate_models(results, X_val, y_val)
            
            # Save models
            self.save_models()
            
            # Pipeline completed
            total_time = (datetime.now() - start_time).total_seconds()
            
            logger.info("\n" + "="*70)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
            logger.info("="*70)
            logger.info(f"Total execution time: {total_time:.2f} seconds")
            logger.info(f"Experiment: {self.experiment_name}")
            logger.info(f"Best model: {report['best_model']['name']}")
            logger.info(f"Best ROC-AUC: {report['best_model']['metrics']['roc_auc']:.4f}")
            logger.info("="*70)
            
            return report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise


def main():
    """Main function to run the pipeline."""
    # Create pipeline
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    pipeline = CreditDefaultPipeline(config_path=config_path)
    
    # Run complete pipeline
    report = pipeline.run_complete_pipeline()
    
    return report


if __name__ == "__main__":
    main()