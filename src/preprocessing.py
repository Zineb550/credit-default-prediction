"""
Data Preprocessing Module
Handles data loading, cleaning, and preprocessing for credit default prediction.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import logging
from typing import Tuple, Dict, Any
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Handles all data preprocessing operations."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the preprocessor with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = None
        self.imputer = None
        self.feature_names = None
        
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from CSV file."""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Drop the index column if it exists
        if df.columns[0] == 'Unnamed: 0' or df.iloc[:, 0].name == '':
            df = df.iloc[:, 1:]
        
        logger.info(f"Data loaded. Shape: {df.shape}")
        return df
    
    def explore_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform exploratory data analysis."""
        logger.info("Performing exploratory data analysis...")
        
        stats = {
            'shape': df.shape,
            'columns': df.columns.tolist(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'dtypes': df.dtypes.to_dict(),
            'target_distribution': df[self.config['data']['target_column']].value_counts().to_dict(),
            'target_percentage': (df[self.config['data']['target_column']].value_counts() / len(df) * 100).to_dict(),
            'numerical_stats': df.describe().to_dict()
        }
        
        logger.info(f"Dataset shape: {stats['shape']}")
        logger.info(f"Target distribution: {stats['target_distribution']}")
        logger.info(f"Missing values: {stats['missing_values']}")
        
        return stats
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'median') -> pd.DataFrame:
        """Handle missing values in the dataset."""
        logger.info(f"Handling missing values with strategy: {strategy}")
        
        # Separate features and target
        target_col = self.config['data']['target_column']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Remove rows with missing target
        mask = y.notna()
        X = X[mask]
        y = y[mask]
        
        # Handle missing values in features
        if strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            self.imputer = SimpleImputer(strategy=strategy)
        
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Combine back with target
        df_clean = pd.concat([X_imputed, y], axis=1)
        
        logger.info(f"Missing values handled. New shape: {df_clean.shape}")
        return df_clean
    
    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 3.0) -> pd.DataFrame:
        """Handle outliers in numerical features."""
        logger.info(f"Handling outliers using {method} method")
        
        target_col = self.config['data']['target_column']
        numerical_cols = [col for col in df.columns if col != target_col and df[col].dtype in ['int64', 'float64']]
        
        df_clean = df.copy()
        
        if method == 'iqr':
            for col in numerical_cols:
                Q1 = df_clean[col].quantile(0.25)
                Q3 = df_clean[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                # Cap outliers instead of removing them
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        elif method == 'zscore':
            for col in numerical_cols:
                mean = df_clean[col].mean()
                std = df_clean[col].std()
                lower_bound = mean - threshold * std
                upper_bound = mean + threshold * std
                df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
        
        logger.info("Outliers handled")
        return df_clean
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features for better prediction."""
        logger.info("Creating advanced features...")
        
        df_features = df.copy()
        
        # Utilization and debt features
        df_features['TotalPastDue'] = (
            df_features['NumberOfTime30-59DaysPastDueNotWorse'] +
            df_features['NumberOfTime60-89DaysPastDueNotWorse'] +
            df_features['NumberOfTimes90DaysLate']
        )
        
        df_features['HasPastDue'] = (df_features['TotalPastDue'] > 0).astype(int)
        df_features['SeverePastDue'] = (df_features['NumberOfTimes90DaysLate'] > 0).astype(int)
        
        # Income and debt features
        df_features['IncomeToDebt'] = df_features['MonthlyIncome'] / (df_features['DebtRatio'] + 1e-10)
        df_features['IncomePerDependent'] = df_features['MonthlyIncome'] / (df_features['NumberOfDependents'] + 1)
        df_features['LoansPerIncome'] = df_features['NumberOfOpenCreditLinesAndLoans'] / (df_features['MonthlyIncome'] + 1)
        
        # Age-related features
        df_features['AgeGroup'] = pd.cut(df_features['age'], bins=[0, 25, 35, 50, 65, 100], 
                                          labels=['Young', 'Adult', 'MiddleAge', 'Senior', 'Elderly'])
        df_features['AgeGroup'] = df_features['AgeGroup'].astype('category').cat.codes
        
        df_features['IsYoung'] = (df_features['age'] < 30).astype(int)
        df_features['IsSenior'] = (df_features['age'] >= 65).astype(int)
        
        # Credit utilization risk
        df_features['HighUtilization'] = (df_features['RevolvingUtilizationOfUnsecuredLines'] > 0.8).astype(int)
        df_features['DebtRisk'] = df_features['DebtRatio'] * df_features['HasPastDue']
        
        # Credit history features
        df_features['LoansToRealEstate'] = df_features['NumberOfOpenCreditLinesAndLoans'] / (df_features['NumberRealEstateLoansOrLines'] + 1)
        df_features['HasRealEstate'] = (df_features['NumberRealEstateLoansOrLines'] > 0).astype(int)
        
        # Risk score combinations
        df_features['RiskScore1'] = (
            df_features['DebtRatio'] * 0.3 +
            df_features['RevolvingUtilizationOfUnsecuredLines'] * 0.3 +
            df_features['TotalPastDue'] * 0.4
        )
        
        df_features['RiskScore2'] = (
            df_features['HasPastDue'] * 0.4 +
            df_features['HighUtilization'] * 0.3 +
            (df_features['NumberOfOpenCreditLinesAndLoans'] > 10).astype(int) * 0.3
        )
        
        # Interaction features (if enabled in config)
        if self.config['features'].get('create_interaction_features', True):
            df_features['Age_DebtRatio'] = df_features['age'] * df_features['DebtRatio']
            df_features['Income_Utilization'] = df_features['MonthlyIncome'] * df_features['RevolvingUtilizationOfUnsecuredLines']
            df_features['PastDue_DebtRatio'] = df_features['TotalPastDue'] * df_features['DebtRatio']
        
        logger.info(f"Feature engineering completed. New shape: {df_features.shape}")
        return df_features
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        logger.info("Scaling features...")
        
        scaling_method = self.config['features'].get('scaling_method', 'standard')
        
        if fit:
            if scaling_method == 'standard':
                self.scaler = StandardScaler()
            elif scaling_method == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaling_method == 'robust':
                self.scaler = RobustScaler()
            
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        return X_scaled
    
    def handle_imbalance(self, X: pd.DataFrame, y: pd.Series, method: str = 'smote') -> Tuple[pd.DataFrame, pd.Series]:
        """Handle class imbalance using various techniques."""
        logger.info(f"Handling class imbalance using {method}...")
        logger.info(f"Original class distribution: {y.value_counts().to_dict()}")
        
        if method == 'smote':
            sampler = SMOTE(random_state=self.config['data']['random_state'], k_neighbors=5)
        elif method == 'adasyn':
            sampler = ADASYN(random_state=self.config['data']['random_state'])
        elif method == 'random_oversample':
            sampler = RandomOverSampler(random_state=self.config['data']['random_state'])
        elif method == 'random_undersample':
            sampler = RandomUnderSampler(random_state=self.config['data']['random_state'])
        elif method == 'smote_tomek':
            sampler = SMOTETomek(random_state=self.config['data']['random_state'])
        else:
            logger.warning(f"Unknown method {method}, using SMOTE")
            sampler = SMOTE(random_state=self.config['data']['random_state'])
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)

        # Limit resampled data to prevent overfitting
        if len(X_resampled) > 50000:
            from sklearn.utils import resample
            logger.info(f"Reducing resampled data from {len(X_resampled)} to 50000 samples to prevent overfitting")
            X_resampled, y_resampled = resample(
                X_resampled, y_resampled, 
                n_samples=50000, 
                random_state=self.config['data']['random_state'], 
                stratify=y_resampled
            )

        # Log AFTER the reduction
        logger.info(f"Resampled class distribution: {pd.Series(y_resampled).value_counts().to_dict()}")
        logger.info(f"New shape: {X_resampled.shape}")
        
        return pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled)
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split data into train and validation sets."""
        logger.info("Splitting data into train and validation sets...")
        
        target_col = self.config['data']['target_column']
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        test_size = self.config['data'].get('validation_size', 0.2)
        random_state = self.config['data']['random_state']
        stratify = y if self.config['training']['stratified'] else None
        
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify
        )
        
        logger.info(f"Train set: {X_train.shape}, Validation set: {X_val.shape}")
        logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
        logger.info(f"Validation target distribution: {y_val.value_counts().to_dict()}")
        
        return X_train, X_val, y_train, y_val
    
    def preprocess_pipeline(self, df: pd.DataFrame, is_training: bool = True) -> Tuple:
        """Complete preprocessing pipeline."""
        logger.info("=" * 50)
        logger.info("Starting preprocessing pipeline...")
        logger.info("=" * 50)
        
        # Explore data
        stats = self.explore_data(df)
        
        # Handle missing values
        df_clean = self.handle_missing_values(df, strategy='median')
        
        # Handle outliers
        df_clean = self.handle_outliers(df_clean, method='iqr')
        
        # Create features
        df_features = self.create_features(df_clean)
        
        if is_training:
            # Split data
            X_train, X_val, y_train, y_val = self.split_data(df_features)
            
            # Scale features
            X_train_scaled = self.scale_features(X_train, fit=True)
            X_val_scaled = self.scale_features(X_val, fit=False)
            
            # Handle imbalance on training data only
            if self.config['training'].get('handle_imbalance', True):
                method = self.config['training'].get('imbalance_method', 'smote')
                X_train_resampled, y_train_resampled = self.handle_imbalance(X_train_scaled, y_train, method)
            else:
                X_train_resampled, y_train_resampled = X_train_scaled, y_train
            
            self.feature_names = X_train_resampled.columns.tolist()
            
            logger.info("=" * 50)
            logger.info("Preprocessing completed successfully!")
            logger.info("=" * 50)
            
            return X_train_resampled, X_val_scaled, y_train_resampled, y_val, stats
        else:
            # For test/prediction data
            target_col = self.config['data']['target_column']
            if target_col in df_features.columns:
                X = df_features.drop(columns=[target_col])
                y = df_features[target_col]
            else:
                X = df_features
                y = None
            
            X_scaled = self.scale_features(X, fit=False)
            
            return X_scaled, y
    
    def save_preprocessor(self, path: str):
        """Save the preprocessor state."""
        import joblib
        preprocessor_state = {
            'scaler': self.scaler,
            'imputer': self.imputer,
            'feature_names': self.feature_names,
            'config': self.config
        }
        joblib.dump(preprocessor_state, path)
        logger.info(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: str):
        """Load the preprocessor state."""
        import joblib
        preprocessor_state = joblib.load(path)
        self.scaler = preprocessor_state['scaler']
        self.imputer = preprocessor_state['imputer']
        self.feature_names = preprocessor_state['feature_names']
        self.config = preprocessor_state['config']
        logger.info(f"Preprocessor loaded from {path}")


if __name__ == "__main__":
    # Test the preprocessor
    preprocessor = DataPreprocessor()
    df = preprocessor.load_data("data/raw/cs-training.csv")
    X_train, X_val, y_train, y_val, stats = preprocessor.preprocess_pipeline(df, is_training=True)
    print(f"\nFinal shapes:")
    print(f"X_train: {X_train.shape}")
    print(f"X_val: {X_val.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"y_val: {y_val.shape}")
