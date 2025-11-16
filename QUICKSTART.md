# Quick Start Guide - Credit Default Prediction

## 🚀 Get Started in 5 Minutes

### Step 1: Setup Environment
```bash
# Make setup script executable (if not already)
chmod +x setup.sh

# Run setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Step 2: Prepare Data
```bash
# Place your training data
cp your-training-data.csv data/raw/cs-training.csv

# Verify data format (should have these columns):
# - SeriousDlqin2yrs (target)
# - RevolvingUtilizationOfUnsecuredLines
# - age
# - DebtRatio
# - MonthlyIncome
# - NumberOfOpenCreditLinesAndLoans
# ... and other features
```

### Step 3: Train Models
```bash
# Navigate to src directory
cd src

# Run training pipeline
python train.py

# This will:
# ✓ Load and preprocess data
# ✓ Engineer features
# ✓ Train multiple models
# ✓ Tune hyperparameters
# ✓ Create ensembles
# ✓ Evaluate and save results
```

### Step 4: View Results
```bash
# Check trained models
ls -lh models/

# View evaluation plots
open results/*_roc_curves.png
open results/*_confusion_matrices.png

# Read evaluation report
cat results/*_evaluation_report.json

# Check training logs
tail -f logs/training.log
```

### Step 5: Make Predictions
```bash
# Run example prediction script
cd examples
python predict.py

# Or use in your own code:
```

```python
import joblib
import pandas as pd
from src.preprocessing import DataPreprocessor

# Load model and preprocessor
model = joblib.load('models/best_model.pkl')
preprocessor_state = joblib.load('models/preprocessor.pkl')

# Load your test data
df_test = pd.read_csv('your_test_data.csv')

# Preprocess
preprocessor = DataPreprocessor()
preprocessor.scaler = preprocessor_state['scaler']
preprocessor.imputer = preprocessor_state['imputer']
X_test, _ = preprocessor.preprocess_pipeline(df_test, is_training=False)

# Predict
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print(f"Predicted defaults: {predictions.sum()}")
```

## ⚙️ Configuration

### Customize Training Parameters

Edit `config/config.yaml`:

```yaml
# Example: Change models to train
training:
  models:
    - "logistic_regression"
    - "random_forest"
    - "xgboost"
    # Add or remove models as needed

# Example: Change hyperparameter tuning
training:
  tune_hyperparameters: true
  tuning_method: "random_search"  # or "grid_search"
  n_iter: 100  # Increase for better results (but slower)

# Example: Change class imbalance handling
training:
  handle_imbalance: true
  imbalance_method: "smote"  # or "adasyn", "random_oversample"
```

## 📊 Expected Training Time

On a modern machine (4 cores, 16GB RAM):
- Data preprocessing: ~30 seconds
- Base model training: ~2-5 minutes
- Hyperparameter tuning: ~10-30 minutes (depending on n_iter)
- Ensemble creation: ~2-5 minutes
- **Total: ~15-40 minutes**

To speed up:
1. Set `tune_hyperparameters: false` in config
2. Reduce `n_iter` value
3. Train fewer models

## 🎯 Key Performance Indicators

After training, check these metrics:

1. **ROC-AUC** (primary metric):
   - Target: > 0.85
   - Excellent: > 0.90

2. **Precision-Recall Balance**:
   - For lending: Higher precision preferred (avoid false approvals)
   - Adjust threshold based on business needs

3. **Cost-Sensitive Metrics**:
   - False Negative Cost: $5
   - False Positive Cost: $1
   - Minimize total cost

## 🔧 Troubleshooting

### Issue: Module not found
```bash
# Make sure you're in the project root directory
cd credit_default_prediction

# And virtual environment is activated
source venv/bin/activate
```

### Issue: CUDA/GPU errors (XGBoost/LightGBM)
```bash
# Install CPU-only versions
pip install xgboost --upgrade --no-cache-dir
pip install lightgbm --upgrade --no-cache-dir
```

### Issue: Memory errors
```bash
# Reduce data size or batch processing in config
# Or increase system swap space
```

### Issue: Training too slow
```yaml
# In config/config.yaml, set:
training:
  tune_hyperparameters: false
  models:
    - "xgboost"  # Train only one fast model
```

## 📈 Improving Model Performance

1. **Feature Engineering**: Add domain-specific features in `preprocessing.py`
2. **Hyperparameter Tuning**: Increase `n_iter` in config
3. **Ensemble Methods**: Enable stacking, voting in config
4. **Class Balancing**: Try different methods (SMOTE, ADASYN)
5. **Threshold Optimization**: Use business cost parameters

## 💡 Tips for Best Results

1. **Data Quality**: 
   - Remove duplicate records
   - Handle missing values properly
   - Check for data leakage

2. **Feature Selection**:
   - Use feature importance plots
   - Remove highly correlated features
   - Create domain-specific features

3. **Model Selection**:
   - Start with simple models (Logistic Regression)
   - Progress to complex models (XGBoost, LightGBM)
   - Use ensembles for best results

4. **Validation**:
   - Use stratified k-fold CV
   - Monitor train-val gap (overfitting)
   - Test on completely unseen data

## 📞 Need Help?

- Check `README.md` for detailed documentation
- Review code comments in source files
- Check logs in `logs/training.log`
- Create an issue on GitHub

## 🎓 Learning Resources

- [Hands-On Machine Learning](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Imbalanced-learn Tutorial](https://imbalanced-learn.org/)

---

**Good luck with your credit default prediction project! 🚀**
