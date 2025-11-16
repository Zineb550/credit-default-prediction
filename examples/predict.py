"""
Example Script: Make Predictions with Trained Model
This script demonstrates how to load a trained model and make predictions.
"""

import sys
import os
import pandas as pd
import numpy as np
import joblib

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.preprocessing import DataPreprocessor
from src.utils import load_model, load_config


def load_trained_pipeline(model_path: str = "models/best_model.pkl",
                          preprocessor_path: str = "models/preprocessor.pkl"):
    """Load trained model and preprocessor."""
    print("Loading trained model and preprocessor...")
    
    model = load_model(model_path)
    preprocessor_state = joblib.load(preprocessor_path)
    
    preprocessor = DataPreprocessor()
    preprocessor.scaler = preprocessor_state['scaler']
    preprocessor.imputer = preprocessor_state['imputer']
    preprocessor.feature_names = preprocessor_state['feature_names']
    preprocessor.config = preprocessor_state['config']
    
    print("✓ Model and preprocessor loaded successfully!")
    return model, preprocessor


def preprocess_new_data(df: pd.DataFrame, preprocessor: DataPreprocessor) -> pd.DataFrame:
    """Preprocess new data using trained preprocessor."""
    print("\nPreprocessing new data...")
    
    # Handle missing values
    df_clean = preprocessor.handle_missing_values(df, strategy='median')
    
    # Handle outliers
    df_clean = preprocessor.handle_outliers(df_clean, method='iqr')
    
    # Create features
    df_features = preprocessor.create_features(df_clean)
    
    # Remove target if it exists
    target_col = preprocessor.config['data']['target_column']
    if target_col in df_features.columns:
        X = df_features.drop(columns=[target_col])
        y = df_features[target_col]
    else:
        X = df_features
        y = None
    
    # Scale features
    X_scaled = preprocessor.scale_features(X, fit=False)
    
    print("✓ Preprocessing completed!")
    return X_scaled, y


def make_predictions(model, X: pd.DataFrame, threshold: float = 0.5) -> tuple:
    """Make predictions using trained model."""
    print(f"\nMaking predictions with threshold={threshold}...")
    
    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        y_proba = model.predict_proba(X)[:, 1]
    else:
        y_proba = model.predict(X)
    
    # Apply threshold
    y_pred = (y_proba >= threshold).astype(int)
    
    print(f"✓ Predictions completed!")
    print(f"  - Total predictions: {len(y_pred)}")
    print(f"  - Predicted defaults: {y_pred.sum()} ({y_pred.sum()/len(y_pred)*100:.2f}%)")
    print(f"  - Predicted non-defaults: {(1-y_pred).sum()} ({(1-y_pred).sum()/len(y_pred)*100:.2f}%)")
    
    return y_pred, y_proba


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray):
    """Evaluate predictions if true labels are available."""
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report, confusion_matrix
    )
    
    print("\n" + "="*60)
    print("EVALUATION METRICS")
    print("="*60)
    
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_true, y_proba):.4f}")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['No Default', 'Default']))
    
    print("="*60)


def create_prediction_report(X: pd.DataFrame, y_pred: np.ndarray, 
                            y_proba: np.ndarray, output_path: str = "predictions.csv"):
    """Create and save prediction report."""
    print(f"\nCreating prediction report...")
    
    # Create results dataframe
    results = X.copy()
    results['predicted_default'] = y_pred
    results['default_probability'] = y_proba
    results['risk_level'] = pd.cut(y_proba, 
                                   bins=[0, 0.3, 0.6, 1.0],
                                   labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    # Save to CSV
    results.to_csv(output_path, index=False)
    print(f"✓ Prediction report saved to {output_path}")
    
    # Display summary
    print("\nRisk Distribution:")
    print(results['risk_level'].value_counts())
    
    return results


def example_single_prediction(model, preprocessor):
    """Example: Predict for a single customer."""
    print("\n" + "="*60)
    print("EXAMPLE: Single Customer Prediction")
    print("="*60)
    
    # Create example customer data
    customer_data = {
        'SeriousDlqin2yrs': [0],  # Unknown, will be removed during preprocessing
        'RevolvingUtilizationOfUnsecuredLines': [0.5],
        'age': [45],
        'NumberOfTime30-59DaysPastDueNotWorse': [0],
        'DebtRatio': [0.3],
        'MonthlyIncome': [5000],
        'NumberOfOpenCreditLinesAndLoans': [8],
        'NumberOfTimes90DaysLate': [0],
        'NumberRealEstateLoansOrLines': [1],
        'NumberOfTime60-89DaysPastDueNotWorse': [0],
        'NumberOfDependents': [2]
    }
    
    df_customer = pd.DataFrame(customer_data)
    
    print("\nCustomer Profile:")
    for col, val in customer_data.items():
        if col != 'SeriousDlqin2yrs':
            print(f"  {col}: {val[0]}")
    
    # Preprocess
    X_customer, _ = preprocess_new_data(df_customer, preprocessor)
    
    # Predict
    y_pred, y_proba = make_predictions(model, X_customer, threshold=0.5)
    
    print(f"\nPrediction Result:")
    print(f"  Default Risk: {'HIGH ⚠️' if y_pred[0] == 1 else 'LOW ✓'}")
    print(f"  Default Probability: {y_proba[0]:.2%}")
    
    if y_proba[0] < 0.3:
        risk_level = "Low Risk - Approve loan"
    elif y_proba[0] < 0.6:
        risk_level = "Medium Risk - Manual review recommended"
    else:
        risk_level = "High Risk - Reject loan"
    
    print(f"  Recommendation: {risk_level}")
    print("="*60)


def main():
    """Main function to demonstrate prediction workflow."""
    print("\n" + "="*60)
    print("CREDIT DEFAULT PREDICTION - INFERENCE EXAMPLE")
    print("="*60)
    
    # Load trained model and preprocessor
    model, preprocessor = load_trained_pipeline()
    
    # Example 1: Single customer prediction
    example_single_prediction(model, preprocessor)
    
    # Example 2: Batch predictions (if you have test data)
    try:
        print("\n" + "="*60)
        print("EXAMPLE: Batch Predictions")
        print("="*60)
        
        # Load test data
        test_data_path = "data/raw/cs-training.csv"  # Replace with your test data
        df_test = pd.read_csv(test_data_path)
        
        # Take a small sample for demonstration
        df_sample = df_test.sample(n=min(100, len(df_test)), random_state=42)
        
        print(f"\nProcessing {len(df_sample)} samples...")
        
        # Preprocess
        X_test, y_test = preprocess_new_data(df_sample, preprocessor)
        
        # Predict
        y_pred, y_proba = make_predictions(model, X_test, threshold=0.5)
        
        # Evaluate (if true labels available)
        if y_test is not None:
            evaluate_predictions(y_test, y_pred, y_proba)
        
        # Create report
        results = create_prediction_report(df_sample, y_pred, y_proba, 
                                          output_path="results/predictions_example.csv")
        
        print("\n✓ Batch prediction completed successfully!")
        
    except FileNotFoundError:
        print("\nTest data not found. Skipping batch prediction example.")
        print("To run batch predictions, place your test data in 'data/raw/'")
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("="*60)


if __name__ == "__main__":
    main()
