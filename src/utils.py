"""
Utility Functions
Helper functions for the credit default prediction project.
"""

import os
import json
import yaml
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def setup_logging(log_dir: str = "logs", log_file: str = "app.log"):
    """Setup logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file)),
            logging.StreamHandler()
        ]
    )


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, output_path: str):
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def save_json(data: Dict, output_path: str):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)


def load_json(file_path: str) -> Dict:
    """Load data from JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def save_model(model: Any, output_path: str):
    """Save model using joblib."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    logger.info(f"Model saved to {output_path}")


def load_model(model_path: str) -> Any:
    """Load model using joblib."""
    model = joblib.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return model


def create_directory_structure(base_dir: str = "."):
    """Create complete directory structure for the project."""
    directories = [
        "data/raw",
        "data/processed",
        "models",
        "logs",
        "results",
        "notebooks",
        "tests",
        "config",
        "src"
    ]
    
    for directory in directories:
        path = os.path.join(base_dir, directory)
        os.makedirs(path, exist_ok=True)
    
    logger.info("Directory structure created successfully")


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame"):
    """Print comprehensive information about a dataframe."""
    print(f"\n{'='*50}")
    print(f"{name} Information")
    print(f"{'='*50}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nData Types:")
    print(df.dtypes)
    print(f"\nMissing Values:")
    print(df.isnull().sum())
    print(f"\nBasic Statistics:")
    print(df.describe())
    print(f"{'='*50}\n")


def calculate_class_weights(y: pd.Series) -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weights = dict(zip(classes, weights))
    
    logger.info(f"Class weights: {class_weights}")
    return class_weights


def plot_class_distribution(y: pd.Series, title: str = "Class Distribution",
                           output_path: str = None):
    """Plot class distribution."""
    plt.figure(figsize=(10, 6))
    
    counts = y.value_counts()
    percentages = y.value_counts(normalize=True) * 100
    
    bars = plt.bar(counts.index, counts.values)
    
    # Add percentage labels on bars
    for i, (bar, count, pct) in enumerate(zip(bars, counts.values, percentages.values)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold')
    
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks([0, 1], ['No Default', 'Default'])
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Class distribution plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_correlation_matrix(df: pd.DataFrame, title: str = "Correlation Matrix",
                           output_path: str = None, figsize: tuple = (12, 10)):
    """Plot correlation matrix heatmap."""
    plt.figure(figsize=figsize)
    
    # Calculate correlation matrix
    corr = df.corr()
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Plot heatmap
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Correlation matrix saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def detect_outliers_iqr(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.Series:
    """Detect outliers using IQR method."""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - threshold * IQR
    upper_bound = Q3 + threshold * IQR
    
    outliers = (df[column] < lower_bound) | (df[column] > upper_bound)
    
    logger.info(f"Outliers in {column}: {outliers.sum()} ({outliers.sum()/len(df)*100:.2f}%)")
    
    return outliers


def generate_eda_report(df: pd.DataFrame, output_dir: str = "results/eda"):
    """Generate comprehensive EDA report."""
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Generating EDA report...")
    
    # Basic statistics
    stats = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'duplicates': int(df.duplicated().sum()),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    # Save statistics
    save_json(stats, os.path.join(output_dir, 'basic_statistics.json'))
    
    # Numerical columns analysis
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numerical_cols:
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        df[col].hist(bins=50)
        plt.title(f'{col} - Distribution')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        
        plt.subplot(1, 3, 2)
        df[col].plot(kind='box')
        plt.title(f'{col} - Box Plot')
        plt.ylabel(col)
        
        plt.subplot(1, 3, 3)
        from scipy import stats
        stats.probplot(df[col].dropna(), dist="norm", plot=plt)
        plt.title(f'{col} - Q-Q Plot')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{col}_analysis.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"EDA report saved to {output_dir}")


def format_time(seconds: float) -> str:
    """Format time in seconds to human-readable format."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


def print_model_summary(model_name: str, metrics: Dict, training_time: float):
    """Print formatted model summary."""
    print(f"\n{'='*60}")
    print(f"MODEL: {model_name}")
    print(f"{'='*60}")
    print(f"Training Time: {format_time(training_time)}")
    print(f"\nMetrics:")
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric_name}: {value:.4f}")
    print(f"{'='*60}\n")


def create_submission_file(predictions: np.ndarray, ids: np.ndarray,
                          output_path: str = "submission.csv"):
    """Create submission file for competition."""
    submission = pd.DataFrame({
        'Id': ids,
        'Prediction': predictions
    })
    
    submission.to_csv(output_path, index=False)
    logger.info(f"Submission file saved to {output_path}")


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality and return issues."""
    issues = {
        'missing_values': {},
        'duplicate_rows': 0,
        'constant_columns': [],
        'high_cardinality_columns': [],
        'outliers': {}
    }
    
    # Missing values
    missing = df.isnull().sum()
    issues['missing_values'] = missing[missing > 0].to_dict()
    
    # Duplicate rows
    issues['duplicate_rows'] = int(df.duplicated().sum())
    
    # Constant columns
    for col in df.columns:
        if df[col].nunique() == 1:
            issues['constant_columns'].append(col)
    
    # High cardinality
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() > 50:
            issues['high_cardinality_columns'].append({
                'column': col,
                'unique_values': int(df[col].nunique())
            })
    
    # Outliers in numerical columns
    for col in df.select_dtypes(include=[np.number]).columns:
        outliers = detect_outliers_iqr(df, col)
        if outliers.sum() > 0:
            issues['outliers'][col] = int(outliers.sum())
    
    return issues


def compare_models_visually(results: Dict[str, Dict], output_path: str):
    """Create visual comparison of multiple models."""
    models = list(results.keys())
    metrics = ['roc_auc', 'f1', 'precision', 'recall', 'accuracy']
    
    # Prepare data
    data = {metric: [] for metric in metrics}
    
    for model in models:
        val_metrics = results[model]['val_metrics']
        for metric in metrics:
            data[metric].append(val_metrics.get(metric, 0))
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    # Plot each metric
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        bars = ax.bar(models, data[metric])
        ax.set_title(f'{metric.upper()}', fontweight='bold')
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom', fontsize=8)
    
    # Hide extra subplot
    axes[-1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Model comparison plot saved to {output_path}")


if __name__ == "__main__":
    print("Utility module loaded successfully")
