"""
Save Processed Data
Extracts and saves the processed data used for training.
"""

import os
import sys
import joblib
import pandas as pd

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.preprocessing import DataPreprocessor

# Load the saved preprocessor
preprocessor_path = os.path.join(project_root, 'models', 'preprocessor.pkl')
print(f"Loading preprocessor from {preprocessor_path}...")

preprocessor_state = joblib.load(preprocessor_path)

print(f"\nPreprocessor state loaded!")
print(f"Features used: {len(preprocessor_state['feature_names'])} features")
print(f"Feature names: {preprocessor_state['feature_names'][:10]}...")

# Now load and process the original data
print("\nLoading and processing original data...")

preprocessor = DataPreprocessor()
preprocessor.scaler = preprocessor_state['scaler']
preprocessor.imputer = preprocessor_state['imputer']
preprocessor.feature_names = preprocessor_state['feature_names']
preprocessor.config = preprocessor_state['config']

# Load original data
data_path = os.path.join(project_root, 'data', 'raw', 'cs-training.csv')
df = preprocessor.load_data(data_path)

print("Processing data with the same pipeline...")
X_train, X_val, y_train, y_val, stats = preprocessor.preprocess_pipeline(df, is_training=True)

# Save processed data
output_dir = os.path.join(project_root, 'data', 'processed')
os.makedirs(output_dir, exist_ok=True)

print(f"\nSaving processed data to {output_dir}...")

X_train.to_csv(os.path.join(output_dir, 'X_train_processed.csv'), index=False)
X_val.to_csv(os.path.join(output_dir, 'X_val_processed.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train_processed.csv'), index=False)
y_val.to_csv(os.path.join(output_dir, 'y_val_processed.csv'), index=False)

print("\n✅ Processed data saved successfully!")
print(f"\nData shapes:")
print(f"X_train: {X_train.shape}")
print(f"X_val: {X_val.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_val: {y_val.shape}")

print(f"\nFiles saved:")
print(f"  - {output_dir}/X_train_processed.csv")
print(f"  - {output_dir}/X_val_processed.csv")
print(f"  - {output_dir}/y_train_processed.csv")
print(f"  - {output_dir}/y_val_processed.csv")
