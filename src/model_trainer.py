"""
Model Training Module
Implements multiple ML algorithms with hyperparameter tuning and ensemble methods.
"""

import pandas as pd
import numpy as np
import yaml
import logging
import joblib
from typing import Dict, List, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sklearn models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

# Ensemble methods
from sklearn.ensemble import StackingClassifier

# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, cross_validate
from sklearn.metrics import make_scorer

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve
)

# External libraries
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logging.warning("LightGBM not available")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Handles model training, tuning, and evaluation."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the model trainer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.models = {}
        self.best_models = {}
        self.results = {}
        self.random_state = self.config['data']['random_state']
        
    def get_base_models(self) -> Dict[str, Any]:
        """Get base models with default parameters."""
        base_models = {}
        
        if 'logistic_regression' in self.config['training']['models']:
            base_models['logistic_regression'] = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                class_weight='balanced'
            )
        
        if 'random_forest' in self.config['training']['models']:
            base_models['random_forest'] = RandomForestClassifier(
                random_state=self.random_state,
                n_estimators=100,
                class_weight='balanced',
                n_jobs=-1
            )
        
        if 'xgboost' in self.config['training']['models'] and XGBOOST_AVAILABLE:
            base_models['xgboost'] = xgb.XGBClassifier(
                random_state=self.random_state,
                n_estimators=100,
                eval_metric='logloss',
                use_label_encoder=False
            )
        
        if 'lightgbm' in self.config['training']['models'] and LIGHTGBM_AVAILABLE:
            base_models['lightgbm'] = lgb.LGBMClassifier(
                random_state=self.random_state,
                n_estimators=100,
                is_unbalance=True,
                verbose=-1
            )
        
        if 'svm' in self.config['training']['models']:
            base_models['svm'] = SVC(
                random_state=self.random_state,
                probability=True,
                class_weight='balanced'
            )
        
        if 'knn' in self.config['training']['models']:
            base_models['knn'] = KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            )
        
        if 'neural_network' in self.config['training']['models']:
            base_models['neural_network'] = MLPClassifier(
                random_state=self.random_state,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1
            )
        
        logger.info(f"Initialized {len(base_models)} base models: {list(base_models.keys())}")
        return base_models
    
    def get_param_distributions(self, model_name: str) -> Dict:
        """Get hyperparameter distributions for a model."""
        if model_name in self.config['models']:
            return self.config['models'][model_name]
        return {}
    
    def train_base_model(self, model_name: str, model: Any, X_train: pd.DataFrame, 
                        y_train: pd.Series, X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train a single base model and evaluate it."""
        logger.info(f"\n{'='*50}")
        logger.info(f"Training {model_name}...")
        logger.info(f"{'='*50}")
        
        start_time = datetime.now()
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_proba_train = model.predict_proba(X_train)[:, 1] if hasattr(model, 'predict_proba') else y_pred_train
        y_proba_val = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else y_pred_val
        
        # Calculate metrics
        train_metrics = self.calculate_metrics(y_train, y_pred_train, y_proba_train)
        val_metrics = self.calculate_metrics(y_val, y_pred_val, y_proba_val)
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        results = {
            'model_name': model_name,
            'model': model,
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'training_time': training_time,
            'y_pred_val': y_pred_val,
            'y_proba_val': y_proba_val
        }
        
        logger.info(f"{model_name} - Training completed in {training_time:.2f}s")
        logger.info(f"Train ROC-AUC: {train_metrics['roc_auc']:.4f}, Val ROC-AUC: {val_metrics['roc_auc']:.4f}")
        logger.info(f"Val Precision: {val_metrics['precision']:.4f}, Val Recall: {val_metrics['recall']:.4f}")
        
        return results
    
    def tune_hyperparameters(self, model_name: str, base_model: Any, 
                            X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Tune hyperparameters using RandomizedSearchCV or GridSearchCV."""
        logger.info(f"\nTuning hyperparameters for {model_name}...")
        
        param_dist = self.get_param_distributions(model_name)
        if not param_dist:
            logger.warning(f"No hyperparameter distribution found for {model_name}")
            return base_model
        
        tuning_method = self.config['training'].get('tuning_method', 'random_search')
        cv_folds = self.config['training'].get('cv_folds', 5)
        scoring_metric = self.config['training'].get('scoring_metric', 'roc_auc')
        n_iter = self.config['training'].get('n_iter', 50)
        
        # Custom scoring for imbalanced datasets
        scoring = {
            'roc_auc': 'roc_auc',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall'
        }
        
        if tuning_method == 'random_search':
            search = RandomizedSearchCV(
                base_model,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=cv_folds,
                scoring=scoring,
                refit=scoring_metric,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=1
            )
        else:  # grid_search
            search = GridSearchCV(
                base_model,
                param_grid=param_dist,
                cv=cv_folds,
                scoring=scoring,
                refit=scoring_metric,
                n_jobs=-1,
                verbose=1
            )
        
        search.fit(X_train, y_train)
        
        logger.info(f"Best parameters for {model_name}: {search.best_params_}")
        logger.info(f"Best CV {scoring_metric}: {search.best_score_:.4f}")
        
        return search.best_estimator_
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate all evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            metrics['pr_auc'] = average_precision_score(y_true, y_proba)
        
        return metrics
    
    def train_all_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                        X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Train all configured models."""
        logger.info("\n" + "="*70)
        logger.info("STARTING MODEL TRAINING PIPELINE")
        logger.info("="*70)
        
        base_models = self.get_base_models()
        results = {}
        
        for model_name, base_model in base_models.items():
            try:
                # Hyperparameter tuning
                if self.config['training'].get('tune_hyperparameters', True):
                    tuned_model = self.tune_hyperparameters(model_name, base_model, X_train, y_train)
                else:
                    tuned_model = base_model
                
                # Train and evaluate
                model_results = self.train_base_model(
                    model_name, tuned_model, X_train, y_train, X_val, y_val
                )
                
                results[model_name] = model_results
                self.best_models[model_name] = tuned_model
                
            except Exception as e:
                logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.results = results
        
        # Print comparison
        self.print_model_comparison(results)
        
        return results
    
    def print_model_comparison(self, results: Dict):
        """Print a comparison table of all models."""
        logger.info("\n" + "="*70)
        logger.info("MODEL COMPARISON")
        logger.info("="*70)
        
        comparison_data = []
        for model_name, result in results.items():
            val_metrics = result['val_metrics']
            comparison_data.append({
                'Model': model_name,
                'ROC-AUC': f"{val_metrics['roc_auc']:.4f}",
                'Precision': f"{val_metrics['precision']:.4f}",
                'Recall': f"{val_metrics['recall']:.4f}",
                'F1-Score': f"{val_metrics['f1']:.4f}",
                'Accuracy': f"{val_metrics['accuracy']:.4f}",
                'Time (s)': f"{result['training_time']:.2f}"
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        print("\n" + df_comparison.to_string(index=False))
        print("\n")
        
        # Find best model
        best_model_name = max(results.keys(), 
                             key=lambda x: results[x]['val_metrics']['roc_auc'])
        logger.info(f"🏆 Best Model: {best_model_name} "
                   f"(ROC-AUC: {results[best_model_name]['val_metrics']['roc_auc']:.4f})")
    
    def create_voting_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                              X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Create a voting ensemble from best models."""
        logger.info("\n" + "="*50)
        logger.info("Creating Voting Ensemble...")
        logger.info("="*50)
        
        # Select top models based on validation ROC-AUC
        top_n = min(5, len(self.best_models))
        top_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['val_metrics']['roc_auc'],
            reverse=True
        )[:top_n]
        
        estimators = [(name, self.best_models[name]) for name, _ in top_models]
        
        # Hard voting
        voting_hard = VotingClassifier(estimators=estimators, voting='hard', n_jobs=-1)
        voting_hard.fit(X_train, y_train)
        
        # Soft voting
        voting_soft = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        voting_soft.fit(X_train, y_train)
        
        # Evaluate both
        results_hard = self.train_base_model('voting_hard', voting_hard, X_train, y_train, X_val, y_val)
        results_soft = self.train_base_model('voting_soft', voting_soft, X_train, y_train, X_val, y_val)
        
        self.results['voting_hard'] = results_hard
        self.results['voting_soft'] = results_soft
        self.best_models['voting_hard'] = voting_hard
        self.best_models['voting_soft'] = voting_soft
        
        return {'voting_hard': results_hard, 'voting_soft': results_soft}
    
    def create_stacking_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Create a stacking ensemble."""
        logger.info("\n" + "="*50)
        logger.info("Creating Stacking Ensemble...")
        logger.info("="*50)
        
        # Select diverse models for stacking
        estimators = []
        for name, model in self.best_models.items():
            if name not in ['voting_hard', 'voting_soft', 'stacking']:
                estimators.append((name, model))
        
        # Use Logistic Regression as meta-learner
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=self.random_state, max_iter=1000),
            cv=5,
            n_jobs=-1
        )
        
        stacking.fit(X_train, y_train)
        
        results_stacking = self.train_base_model('stacking', stacking, X_train, y_train, X_val, y_val)
        
        self.results['stacking'] = results_stacking
        self.best_models['stacking'] = stacking
        
        return {'stacking': results_stacking}
    
    def create_bagging_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Create bagging ensembles."""
        logger.info("\n" + "="*50)
        logger.info("Creating Bagging Ensembles...")
        logger.info("="*50)
        
        # Get the best single model
        best_model_name = max(
            [k for k in self.results.keys() if k not in ['voting_hard', 'voting_soft', 'stacking']],
            key=lambda x: self.results[x]['val_metrics']['roc_auc']
        )
        best_model = self.best_models[best_model_name]
        
        # Create bagging ensemble
        bagging = BaggingClassifier(
            estimator=best_model,
            n_estimators=10,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        bagging.fit(X_train, y_train)
        
        results_bagging = self.train_base_model('bagging', bagging, X_train, y_train, X_val, y_val)
        
        self.results['bagging'] = results_bagging
        self.best_models['bagging'] = bagging
        
        return {'bagging': results_bagging}
    
    def create_boosting_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                                X_val: pd.DataFrame, y_val: pd.Series) -> Dict:
        """Create AdaBoost ensemble."""
        logger.info("\n" + "="*50)
        logger.info("Creating Boosting Ensemble...")
        logger.info("="*50)
        
        # AdaBoost with Decision Tree
        boosting = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3, random_state=self.random_state),
            n_estimators=100,
            random_state=self.random_state,
            algorithm='SAMME'
        )
        
        boosting.fit(X_train, y_train)
        
        results_boosting = self.train_base_model('boosting', boosting, X_train, y_train, X_val, y_val)
        
        self.results['boosting'] = results_boosting
        self.best_models['boosting'] = boosting
        
        return {'boosting': results_boosting}
    
    def train_ensemble_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series):
        """Train all ensemble models."""
        logger.info("\n" + "="*70)
        logger.info("TRAINING ENSEMBLE MODELS")
        logger.info("="*70)
        
        if self.config['training'].get('use_ensemble', True):
            ensemble_methods = self.config['training'].get('ensemble_methods', [])
            
            if 'voting' in ensemble_methods:
                self.create_voting_ensemble(X_train, y_train, X_val, y_val)
            
            if 'stacking' in ensemble_methods:
                self.create_stacking_ensemble(X_train, y_train, X_val, y_val)
            
            if 'bagging' in ensemble_methods:
                self.create_bagging_ensemble(X_train, y_train, X_val, y_val)
            
            if 'boosting' in ensemble_methods:
                self.create_boosting_ensemble(X_train, y_train, X_val, y_val)
            
            # Final comparison with ensemble models
            self.print_model_comparison(self.results)
    
    def get_best_model(self) -> Tuple[str, Any]:
        """Get the best performing model."""
        best_model_name = max(self.results.keys(), 
                             key=lambda x: self.results[x]['val_metrics']['roc_auc'])
        return best_model_name, self.best_models[best_model_name]
    
    def save_model(self, model_name: str, model: Any, path: str):
        """Save a trained model."""
        joblib.dump(model, path)
        logger.info(f"Model {model_name} saved to {path}")
    
    def save_all_models(self, output_dir: str = "models"):
        """Save all trained models."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for model_name, model in self.best_models.items():
            model_path = f"{output_dir}/{model_name}_model.pkl"
            self.save_model(model_name, model, model_path)
        
        # Save the best model separately
        best_model_name, best_model = self.get_best_model()
        best_model_path = f"{output_dir}/best_model.pkl"
        self.save_model(best_model_name, best_model, best_model_path)
        
        # Save results
        results_path = f"{output_dir}/training_results.pkl"
        joblib.dump(self.results, results_path)
        logger.info(f"All models and results saved to {output_dir}")
    
    def load_model(self, path: str) -> Any:
        """Load a trained model."""
        model = joblib.load(path)
        logger.info(f"Model loaded from {path}")
        return model


if __name__ == "__main__":
    # Test the model trainer
    logger.info("Testing Model Trainer...")
