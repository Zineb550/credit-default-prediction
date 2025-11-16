# Credit Default Prediction - End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Project Overview

This is a comprehensive end-to-end machine learning project that predicts credit default risk for banking institutions. The project implements the complete ML lifecycle including data preprocessing, feature engineering, model training with multiple algorithms, hyperparameter tuning, ensemble methods, model evaluation, and deployment-ready pipeline.

**Business Problem:** Predict who will default on their loans so banks can make better lending decisions, minimize financial risk, and optimize their credit approval process.

**Target Variable:** `SeriousDlqin2yrs` - Binary classification (0: No Default, 1: Default)

## 🌟 Key Features

- **Advanced Data Preprocessing**
  - Missing value imputation (median, KNN)
  - Outlier detection and handling (IQR, Z-score methods)
  - Feature scaling (Standard, MinMax, Robust scalers)
  
- **Sophisticated Feature Engineering**
  - 20+ engineered features including:
    - Risk scores and composite indicators
    - Interaction features
    - Ratio-based features
    - Age and income groupings
    - Credit utilization patterns

- **Multiple ML Algorithms**
  - Logistic Regression
  - Random Forest
  - XGBoost
  - LightGBM
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
  - Neural Networks (MLP)

- **Advanced Ensemble Methods**
  - Voting Classifier (Hard & Soft)
  - Stacking Classifier
  - Bagging
  - Boosting (AdaBoost)

- **Comprehensive Model Evaluation**
  - ROC-AUC, Precision-Recall curves
  - Confusion matrices
  - Cost-sensitive metrics
  - Threshold optimization
  - Feature importance analysis
  - Calibration curves

- **Class Imbalance Handling**
  - SMOTE
  - ADASYN
  - Random Over/Under Sampling
  - SMOTE-Tomek

- **Hyperparameter Tuning**
  - RandomizedSearchCV
  - GridSearchCV
  - 5-fold stratified cross-validation

## 📊 Dataset Information

- **Source:** Credit Scoring Dataset
- **Size:** 150,000+ records
- **Features:** 11 features including:
  - Age
  - Debt Ratio
  - Monthly Income
  - Number of Open Credit Lines
  - Credit Utilization
  - Past Due Indicators
  - Number of Dependents
  - Real Estate Loans

- **Target Distribution:** Highly imbalanced (~6.7% default rate)

## 🏗️ Project Structure

```
credit_default_prediction/
│
├── config/
│   └── config.yaml                 # Central configuration file
│
├── data/
│   ├── raw/                        # Raw datasets
│   │   └── cs-training.csv
│   └── processed/                  # Processed datasets
│
├── src/
│   ├── preprocessing.py            # Data preprocessing module
│   ├── model_trainer.py            # Model training and tuning
│   ├── evaluator.py                # Model evaluation and analysis
│   ├── train.py                    # Main training script
│   └── utils.py                    # Utility functions
│
├── models/                         # Trained models
│   ├── best_model.pkl
│   ├── preprocessor.pkl
│   └── *.pkl                       # Individual model files
│
├── results/                        # Results and visualizations
│   ├── roc_curves.png
│   ├── pr_curves.png
│   ├── confusion_matrices.png
│   └── evaluation_report.json
│
├── logs/                           # Training logs
│   └── training.log
│
├── notebooks/                      # Jupyter notebooks
│
├── tests/                          # Unit tests
│
├── requirements.txt                # Python dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore file
```

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8 or higher
pip or conda package manager
```

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd credit_default_prediction
```

2. **Create virtual environment**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n credit_ml python=3.8
conda activate credit_ml
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify installation**
```bash
python -c "import sklearn, xgboost, lightgbm; print('All packages installed successfully!')"
```

### Data Setup

1. Place your training data in `data/raw/cs-training.csv`
2. The dataset should have the following columns:
   - SeriousDlqin2yrs (target)
   - RevolvingUtilizationOfUnsecuredLines
   - age
   - NumberOfTime30-59DaysPastDueNotWorse
   - DebtRatio
   - MonthlyIncome
   - NumberOfOpenCreditLinesAndLoans
   - NumberOfTimes90DaysLate
   - NumberRealEstateLoansOrLines
   - NumberOfTime60-89DaysPastDueNotWorse
   - NumberOfDependents

## 🎓 Usage

### Training the Model

**Basic training** (uses default configuration):
```bash
cd src
python train.py
```

**Advanced configuration** (modify `config/config.yaml`):
```yaml
training:
  models:
    - "logistic_regression"
    - "random_forest"
    - "xgboost"
    - "lightgbm"
  
  tune_hyperparameters: true
  use_ensemble: true
  handle_imbalance: true
  imbalance_method: "smote"
```

### Training Pipeline Stages

The training pipeline executes the following stages:

1. **Data Loading** - Loads raw training data
2. **Preprocessing** - Cleans data, handles missing values, outliers
3. **Feature Engineering** - Creates 20+ advanced features
4. **Data Splitting** - Stratified train-validation split
5. **Class Balancing** - Applies SMOTE or other techniques
6. **Model Training** - Trains multiple ML algorithms
7. **Hyperparameter Tuning** - Optimizes each model
8. **Ensemble Creation** - Builds voting, stacking, bagging ensembles
9. **Evaluation** - Comprehensive performance analysis
10. **Model Persistence** - Saves best models and artifacts

### Expected Output

After training completes, you'll find:

```
models/
├── best_model.pkl                 # Best performing model
├── logistic_regression_model.pkl
├── random_forest_model.pkl
├── xgboost_model.pkl
├── lightgbm_model.pkl
├── stacking_model.pkl
└── preprocessor.pkl               # Preprocessing pipeline

results/
├── experiment_YYYYMMDD_HHMMSS_roc_curves.png
├── experiment_YYYYMMDD_HHMMSS_pr_curves.png
├── experiment_YYYYMMDD_HHMMSS_confusion_matrices.png
├── experiment_YYYYMMDD_HHMMSS_feature_importance.png
└── experiment_YYYYMMDD_HHMMSS_evaluation_report.json

logs/
└── training.log                   # Detailed training logs
```

## 📈 Model Performance

Expected performance metrics (after hyperparameter tuning):

| Model | ROC-AUC | Precision | Recall | F1-Score |
|-------|---------|-----------|--------|----------|
| Logistic Regression | 0.82-0.84 | 0.15-0.20 | 0.60-0.70 | 0.25-0.30 |
| Random Forest | 0.83-0.85 | 0.16-0.22 | 0.65-0.75 | 0.26-0.32 |
| XGBoost | 0.84-0.86 | 0.17-0.23 | 0.65-0.75 | 0.27-0.34 |
| LightGBM | 0.84-0.87 | 0.17-0.24 | 0.66-0.76 | 0.27-0.35 |
| Stacking Ensemble | 0.85-0.88 | 0.18-0.25 | 0.67-0.77 | 0.28-0.36 |

*Note: Actual performance may vary based on data and hyperparameters*

## ⚙️ Configuration

### Key Configuration Parameters

Edit `config/config.yaml` to customize:

```yaml
# Data preprocessing
features:
  scaling_method: "standard"  # standard, minmax, robust
  create_interaction_features: true

# Training
training:
  cv_folds: 5
  tune_hyperparameters: true
  tuning_method: "random_search"  # random_search, grid_search
  n_iter: 50
  scoring_metric: "roc_auc"
  
  # Class imbalance
  handle_imbalance: true
  imbalance_method: "smote"  # smote, adasyn, random_oversample

# Evaluation
evaluation:
  cost_sensitive_evaluation: true
  false_positive_cost: 1
  false_negative_cost: 5  # FN costs 5x more than FP
```

## 🔬 Advanced Features

### Custom Feature Engineering

The project includes sophisticated feature engineering:

```python
# Risk scores
RiskScore1 = DebtRatio * 0.3 + RevolvingUtilization * 0.3 + TotalPastDue * 0.4

# Income-based features
IncomeToDebt = MonthlyIncome / (DebtRatio + ε)
IncomePerDependent = MonthlyIncome / (NumberOfDependents + 1)

# Credit utilization
HighUtilization = 1 if RevolvingUtilization > 0.8 else 0
```

### Cost-Sensitive Learning

The project implements cost-sensitive evaluation considering:
- **False Positive Cost:** $1 (rejected good customer)
- **False Negative Cost:** $5 (approved bad customer who defaults)

This reflects real-world business impact where missed defaults are costlier.

### Threshold Optimization

Automatically finds the optimal classification threshold by:
1. Testing thresholds from 0.1 to 0.9
2. Calculating cost-sensitive metrics
3. Selecting threshold that minimizes total business cost

## 📊 Visualization Examples

The pipeline generates comprehensive visualizations:

1. **ROC Curves** - Compare all models' discriminative ability
2. **Precision-Recall Curves** - Important for imbalanced data
3. **Confusion Matrices** - Detailed prediction breakdown
4. **Feature Importance** - Top predictive features
5. **Calibration Curves** - Probability calibration assessment
6. **Error Analysis** - Patterns in misclassifications

## 🧪 Testing

Run unit tests:
```bash
pytest tests/
```

Run with coverage:
```bash
pytest tests/ --cov=src --cov-report=html
```

## 📝 Logging

The project maintains detailed logs in `logs/training.log`:

```
2024-01-15 10:30:45 - INFO - Loading data from data/raw/cs-training.csv
2024-01-15 10:30:46 - INFO - Data loaded. Shape: (150000, 12)
2024-01-15 10:30:46 - INFO - Target distribution: {0: 139974, 1: 10026}
2024-01-15 10:30:50 - INFO - Preprocessing completed successfully!
2024-01-15 10:31:00 - INFO - Training logistic_regression...
...
```

## 🚀 Deployment (ZenML Integration)

### ZenML Pipeline Setup

```bash
# Install ZenML
pip install zenml[server]

# Initialize ZenML
zenml init

# Register your pipeline
zenml pipeline run credit_default_pipeline
```

### Create ZenML Pipeline

```python
from zenml import pipeline, step

@step
def preprocess_data() -> ...:
    # Your preprocessing code
    pass

@step
def train_model() -> ...:
    # Your training code
    pass

@pipeline
def credit_default_pipeline():
    X_train, y_train = preprocess_data()
    model = train_model(X_train, y_train)
    return model
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- **Your Name** - Initial work - [YourGitHub](https://github.com/yourusername)

## 🙏 Acknowledgments

- Inspired by Kaggle's "Give Me Some Credit" competition
- Based on best practices from "Hands-On Machine Learning" by Aurélien Géron
- Built following the ML lifecycle framework from INPT coursework

## 📞 Contact

For questions or support:
- Email: your.email@example.com
- GitHub Issues: [Create an issue](https://github.com/yourusername/credit-default-prediction/issues)

## 🔗 Related Resources

- [Hands-On Machine Learning Book](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [ZenML Documentation](https://docs.zenml.io/)

---

**Note:** This project is for educational purposes as part of the INE2-DATA Machine Learning course at INPT, 2025.

Made with ❤️ and lots of ☕
