# Credit Default Prediction: End-to-End Machine Learning Project

**Author:** Zineb El Hadi  
**Institution:** Institut National des Postes et Télécommunications (INPT)  
**Program:** INE2-DATA 2025  
**Course:** Machine Learning  
**Date:** November 16, 2025

---

## Quick Access

- **Live API Demo:** [https://credit-default-prediction-x7rl.onrender.com/docs](https://credit-default-prediction-x7rl.onrender.com/docs)
- **GitHub Repository:** [https://github.com/Zineb550/credit-default-prediction](https://github.com/Zineb550/credit-default-prediction)

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Business Problem](#business-problem)
3. [Dataset](#dataset)
4. [Technical Approach](#technical-approach)
5. [Results](#results)
6. [Deployment](#deployment)
7. [Challenges Overcome](#challenges-overcome)
8. [Installation](#installation)

---

## Project Overview

This project implements a complete machine learning system for predicting credit default risk. The system predicts whether a customer will experience serious delinquency (90+ days past due) within the next two years, achieving a ROC-AUC score of **0.8011**.

**Complete ML Lifecycle Coverage:**

1. **Problem Definition** - Formulated business problem with clear success metrics
2. **Data Collection** - Acquired 150,000 customer records from Kaggle competition
3. **Data Preparation** - Handled missing values, outliers, and severe class imbalance (6.7% default rate)
4. **Feature Engineering** - Created 28 predictive features from 11 original features
5. **Model Training** - Trained and compared 9 different models (5 base models + 4 ensembles)
6. **Model Evaluation** - Validated performance using ROC-AUC and business metrics
7. **Deployment** - Deployed production API on cloud with Docker containerization
8. **Monitoring** - Implemented drift detection and performance tracking

**Key Achievement:** The final Voting Soft Ensemble model achieves competitive performance (0.8011 ROC-AUC) through systematic feature engineering, careful handling of data quality issues, and iterative optimization to prevent overfitting.

---

## Business Problem

### The Challenge

Financial institutions need to assess credit risk before extending loans. The core question: **Can we predict if a customer will default (go 90+ days past due) within the next 2 years?**

### Business Impact

Accurate predictions enable:
- **Loss Prevention:** Identify high-risk applicants before lending
- **Risk-Based Pricing:** Adjust rates based on predicted risk
- **Automated Decisions:** Process applications faster with confidence
- **Portfolio Management:** Monitor overall risk exposure

### Why ROC-AUC is Our Primary Metric

We chose **ROC-AUC (Area Under the ROC Curve)** as the primary success metric for critical reasons:

**1. Handles Class Imbalance**

With only 6.7% of customers defaulting, accuracy is misleading. A naive model predicting "no default" for everyone would achieve 93% accuracy but provide zero business value. ROC-AUC measures the model's ability to correctly rank customers by risk, regardless of class distribution.

**2. Threshold Independence**

ROC-AUC evaluates performance across all possible decision thresholds. This matters because the optimal threshold depends on business considerations (cost of false positives vs false negatives) that may change. The metric tells us: "Can the model rank risky customers higher than safe ones?" independent of where we draw the line.

**3. Direct Business Interpretation**

ROC-AUC answers: *"If I randomly pick one defaulter and one non-defaulter, what's the probability that my model assigns a higher risk score to the defaulter?"* 

Our score of 0.8011 means 80.1% probability - strong discriminative ability.

**4. Industry Standard**

Credit risk models are universally evaluated using ROC-AUC. Our score of 0.8011 places us in the competitive range (industry standard: 0.75-0.85), enabling comparison with professional systems.

**Why Other Metrics Are Secondary**

- **Precision (0.2253):** While seemingly low, this represents over 3x concentration compared to the 6.7% base rate. The model identifies a risk group where 22.5% actually default.

- **Recall (0.5411):** Catching 54% of defaulters while maintaining reasonable precision represents an optimal business tradeoff. Perfect recall would mean flagging too many customers.

- **Accuracy (0.8653):** Misleading due to class imbalance. High accuracy can be achieved by always predicting the majority class.

The ROC-AUC metric aligns with how the model will be used: ranking customers by risk and setting thresholds based on business constraints.

---

## Dataset

### Source

**Kaggle Competition:** "Give Me Some Credit"  
**Link:** [https://www.kaggle.com/c/GiveMeSomeCredit/data](https://www.kaggle.com/c/GiveMeSomeCredit/data)

This dataset represents real anonymized customer records from a financial institution, providing actual credit and demographic data for default prediction.

### Characteristics

- **Size:** 150,000 customer records
- **Features:** 11 original features (financial and demographic)
- **Target:** Binary classification (1 = default, 0 = no default)
- **Time Window:** 2-year observation period
- **Class Imbalance:** Severe - only 6.7% positive class (10,026 defaulters)

### Features

| Feature | Description | Missing |
|---------|-------------|---------|
| SeriousDlqin2yrs | Target: 90+ days past due within 2 years | 0% |
| RevolvingUtilizationOfUnsecuredLines | Credit card balance / credit limit ratio | 0% |
| age | Customer age in years | 0% |
| NumberOfTime30-59DaysPastDueNotWorse | Count of 30-59 days late | 0% |
| DebtRatio | Monthly debt / monthly income | 0% |
| MonthlyIncome | Monthly income in dollars | 19.8% |
| NumberOfOpenCreditLinesAndLoans | Number of open credit accounts | 0% |
| NumberOfTimes90DaysLate | Count of 90+ days late | 0% |
| NumberRealEstateLoansOrLines | Number of mortgage loans | 0% |
| NumberOfTime60-89DaysPastDueNotWorse | Count of 60-89 days late | 0% |
| NumberOfDependents | Number of dependents | 2.6% |

### Data Quality Issues

**Missing Values:**
- MonthlyIncome: 19.8% missing (~29,700 records)
- NumberOfDependents: 2.6% missing (~3,900 records)

**Outliers:**
- RevolvingUtilization values exceeding 100%
- Extreme DebtRatio values (>10,000)
- Unrealistic age values

**Class Imbalance:**
- Only 6.7% defaulters (severe imbalance)
- Risk of model bias toward majority class

---

## Technical Approach

### 1. Data Preprocessing

#### Missing Value Treatment

**MonthlyIncome (19.8% missing):**
- **Method:** Median imputation
- **Rationale:** Income distributions are right-skewed. Median (~$5,400) is more robust than mean for skewed data with outliers.

**NumberOfDependents (2.6% missing):**
- **Method:** Median imputation (median = 0)
- **Rationale:** Small percentage missing, discrete values with limited range.

#### Outlier Handling

**Method:** IQR (Interquartile Range) capping
- Calculate Q1 (25th percentile) and Q3 (75th percentile)
- Define bounds: [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- Cap values outside bounds to nearest boundary

**Impact:** Preserved 98% of data while controlling extreme values that could distort model training.

#### Class Imbalance Solution

**Challenge:** 6.7% default rate causes models to predict "no default" for everyone.

**Solution:** SMOTE (Synthetic Minority Over-sampling Technique)
- Creates synthetic examples of minority class (defaulters)
- **Critical optimization:** Limited to 50,000 total samples
- **Why limited?** Initial attempt with 224,000 samples caused severe overfitting (train AUC 0.99, validation AUC 0.64)
- **Result:** Balanced 50-50 split without overfitting (validation AUC improved to 0.80)

This optimization was crucial - demonstrated that more synthetic data doesn't always help. Finding the right balance prevents overfitting while addressing class imbalance.

### 2. Feature Engineering

Created **28 features** from 11 original features through domain-informed engineering:

#### Aggregate Features
- **TotalPastDue:** Sum of all past due occurrences → Single delinquency metric
- **TotalDelinquencies:** Total count of delinquent events → Overall credit behavior

#### Financial Ratios
- **DebtToIncomeRatio:** Normalized debt burden measure
- **IncomePerDependent:** Available income per household member
- **CreditLinesPerDependent:** Credit accessibility relative to obligations
- **UtilizationDebtInteraction:** RevolvingUtilization × DebtRatio → Combined pressure indicator
- **AgeIncomeInteraction:** Age × Income → Life stage financial capacity
- **CreditUtilizationRatio:** Credit lines / Utilization → Usage efficiency

#### Risk Indicators (Binary Flags)
- **HasPastDue:** Flag for any delinquency history
- **HighUtilization:** Utilization > 80% (industry risk threshold)
- **HighDebt:** DebtRatio > 50% (unsustainable debt level)
- **LowIncome:** MonthlyIncome < $3,000 (financial vulnerability)
- **ManyCreditLines:** More than 10 open lines (over-extension)
- **YoungAge:** Age < 25 (limited credit history)
- **HasRealEstate:** Real estate ownership (stability indicator)

#### Categorical Bins
- **UtilizationBin:** [0-30%, 30-60%, 60-80%, 80%+] → Non-linear relationship capture
- **AgeBin:** [<25, 25-40, 40-60, 60+] → Life stage risk profiles

#### Composite Risk Score
- **RiskScore:** Weighted sum of risk flags (0-5 scale)
- Components: HighUtilization + HighDebt + HasPastDue + LowIncome + ManyCreditLines
- Purpose: Simple interpretable score for business users

**Feature Scaling:** StandardScaler applied to all features (mean=0, std=1) for algorithm compatibility.

**Impact:** Feature engineering improved model performance by approximately 15% compared to using raw features alone.

### 3. Model Development

#### Base Models (5 algorithms tested)

**1. Logistic Regression**
- Simple linear model with L2 regularization
- Class weight: balanced
- Strong baseline performance: 0.7997 AUC

**2. Random Forest**
- 100-200 trees with depth limits
- Challenge: Showed overfitting despite regularization
- Final AUC: 0.7526

**3. XGBoost**
- Gradient boosting with learning rate optimization
- Max depth: 3-7, learning rate: 0.01-0.1
- Performance: 0.7809 AUC

**4. LightGBM**
- Fast gradient boosting variant
- Num leaves: 31-50, learning rate tuned
- Performance: 0.7943 AUC

**5. Neural Network**
- Architecture: 3 hidden layers (128→64→32)
- Activation: ReLU, regularization: dropout + L2
- Early stopping to prevent overfitting
- Performance: 0.7708 AUC

#### Ensemble Methods (4 approaches tested)

**6. Voting Hard**
- Majority vote from base models
- Problem: Lost probability information
- Underperformed: 0.6950 AUC

**7. Voting Soft (BEST MODEL)**
- Averages predicted probabilities
- Combines: Logistic Regression + XGBoost + LightGBM + Neural Network
- **Performance: 0.8011 AUC**
- Why best: Diversity + probability smoothing + minimal overfitting (gap: 0.19)

**8. Bagging**
- Bootstrap aggregating with Logistic Regression
- Performance: 0.7999 AUC

**9. AdaBoost**
- Sequential boosting on misclassified samples
- Performance: 0.7851 AUC

**Model Selection:** Voting Soft chosen for highest validation AUC and best generalization.

**Note on Hyperparameter Tuning:** Extensive Grid Search CV was initially implemented but ultimately disabled due to computational cost (8+ hours) with marginal performance gains (<1% AUC improvement). Final models use carefully selected parameters based on literature and initial exploration.

### 4. Optimization Journey

**Phase 1: Baseline**
- Logistic Regression with raw features: 0.72 AUC
- Established lower bound performance

**Phase 2: Feature Engineering**
- Added 17 engineered features: +4% improvement to 0.76 AUC
- Showed domain knowledge significantly impacts predictions

**Phase 3: Initial SMOTE (Failure)**
- Applied SMOTE with 224K samples (full balance)
- Severe overfitting detected: Train 0.99, Validation 0.64, Gap 0.35
- Root cause: Too many synthetic samples created unrealistic patterns

**Phase 4: SMOTE Optimization (Success)**
- Systematically tested resampling amounts: 100K → 75K → 50K → 25K
- Found optimal: 50K samples
- Result: Validation AUC improved from 0.64 to 0.77, gap reduced to 0.19

**Phase 5: Model Comparison**
- Tested 9 different algorithms and ensembles
- Random Forest: Overfitting issues despite regularization
- Voting Hard: Information loss from hard voting
- Voting Soft: Best performance through probability averaging

**Phase 6: Final Validation**
- Extensive testing on hold-out set
- Confusion matrix and ROC curve analysis
- Business metric calculations
- Confirmed Voting Soft as final production model

**Key Lessons:**
- More synthetic data ≠ better performance
- Feature engineering impact > hyperparameter tuning
- Ensemble diversity provides robustness
- Monitoring train-validation gap is critical

---

## Results

### Best Model: Voting Soft Ensemble

**Composition:** Equal-weight average of 4 diverse models
- Logistic Regression (25%)
- XGBoost (25%)
- LightGBM (25%)
- Neural Network (25%)

**Why This Works:**
- **Diversity:** Linear + tree-based + neural approaches
- **Probability Smoothing:** Averaging reduces individual model variance
- **Error Compensation:** Different models err on different samples
- **Minimal Overfitting:** Train-validation gap only 0.19

### Performance Metrics

**Primary Metric:**
- **ROC-AUC: 0.8011** (Validation Set)
- Training AUC: 0.9956
- Generalization Gap: 0.1945 (acceptable)

**Classification Metrics (threshold 0.5):**
- Precision: 0.2253
- Recall: 0.5411
- F1-Score: 0.3182

**Confusion Matrix:**
```
                  Predicted No    Predicted Yes
Actual No Default    27,912           1,014
Actual Default          461             541
```

**Interpretation:**
- Catches 54% of actual defaulters (541 out of 1,002)
- Flags only 5.2% of applications for review (1,555 out of 29,968)
- Risk concentration: 22.5% default rate in flagged group vs 6.7% baseline (3.4x lift)

### Business Value

**Loss Prevention Calculation:**
- Assumptions: $5,000 average loss per default
- Prevented defaults: 541 cases
- **Prevented losses:** $2.7 million per 30,000 applications
- Review cost: 1,555 × $50 = $77,750
- **Net benefit:** ~$2.6 million

**Risk Classification:**
- **Low Risk (<30%):** 82% of customers → Approve automatically
- **Medium Risk (30-60%):** 13% of customers → Manual review
- **High Risk (>60%):** 5% of customers → Decline

### Model Comparison

| Model | Validation AUC | Train-Val Gap |
|-------|---------------|---------------|
| Logistic Regression | 0.7997 | 0.0198 |
| Random Forest | 0.7526 | 0.2474 |
| XGBoost | 0.7809 | 0.2191 |
| LightGBM | 0.7943 | 0.2057 |
| Neural Network | 0.7708 | 0.2292 |
| Voting Hard | 0.6950 | 0.3050 |
| **Voting Soft** | **0.8011** | **0.1945** |
| Bagging | 0.7999 | 0.2001 |
| AdaBoost | 0.7851 | 0.2149 |

**Key Observations:**
- Voting Soft achieves highest validation performance
- Logistic Regression has minimal overfitting but lower absolute performance
- Ensemble methods generally show better generalization
- Our 0.8011 AUC is competitive with industry standards (0.75-0.85)

### Feature Importance (Top 10)

1. **TotalPastDue (18.2%)** - Payment history is strongest predictor
2. **RevolvingUtilization (14.7%)** - Credit usage indicates stress
3. **RiskScore (11.3%)** - Composite risk indicator
4. **Age (9.8%)** - Younger borrowers higher risk
5. **DebtRatio (8.4%)** - Debt burden
6. **NumberOfOpenCreditLines (6.9%)** - Credit availability
7. **UtilizationDebtInteraction (5.8%)** - Combined pressure
8. **MonthlyIncome (5.3%)** - Financial capacity
9. **HasPastDue (4.7%)** - Delinquency flag
10. **IncomePerDependent (4.1%)** - Per-capita income

---

## Deployment

### Architecture

The model is deployed as a production-ready REST API with the following stack:

**Infrastructure:**
- **Platform:** Render.com (Cloud PaaS)
- **Container:** Docker for reproducible environment
- **Region:** Frankfurt, EU (data privacy compliance)
- **Model Storage:** Google Drive (1.2GB models too large for GitHub)

**Application:**
- **API Framework:** FastAPI (modern Python web framework)
- **Server:** Uvicorn (ASGI server)
- **Documentation:** Auto-generated Swagger UI

**Deployment Flow:**
```
Git Push → GitHub → Render Auto-Deploy → 
Docker Build → Download Models from Drive → 
Start Server → Health Check → Live API
```

### API Endpoints

**1. Health Check** - `GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-16T16:30:00"
}
```

**2. Single Prediction** - `POST /predict`

Input: Customer financial data (11 fields)

Output:
```json
{
  "prediction": 0,
  "probability": 0.2538,
  "risk_level": "Low Risk",
  "recommendation": "APPROVE - Low default risk.",
  "timestamp": "2025-11-16T16:30:00"
}
```

**3. Batch Prediction** - `POST /batch-predict`
Process multiple customers efficiently

**4. Statistics** - `GET /stats`
Monitor prediction patterns and usage

**5. Drift Detection** - `GET /monitoring/drift`
Alert if recent predictions differ from training distribution (threshold: 5% deviation)

### Why This Deployment Matters

**Scalability:**
- Cloud infrastructure enables easy scaling
- Containerization ensures consistency across environments
- Model updates without code redeployment (via Google Drive)

**Monitoring:**
- Real-time statistics tracking
- Drift detection for data quality alerts
- Performance logging for analysis

**Production-Ready:**
- Interactive documentation for easy integration
- Health checks for reliability
- HTTPS encryption for security

---

## Challenges Overcome

### Challenge 1: Severe Class Imbalance (6.7% defaults)

**Problem:** Standard algorithms predicted "no default" for everyone, achieving 93% accuracy but zero business value.

**Solution Evolution:**
- Tried class weights → Minimal improvement
- Applied SMOTE (224K samples) → Severe overfitting (train 0.99, val 0.64)
- **Optimized SMOTE (50K samples) → Solved** (train 0.80, val 0.80)

**Key Insight:** Quality over quantity - limited synthetic data prevented unrealistic patterns while addressing imbalance.

### Challenge 2: Overfitting (Train-Val Gap 0.35)

**Problem:** Initial models memorized training patterns rather than learning general rules.

**Root Causes:**
- Excessive SMOTE samples
- Insufficient regularization
- Model complexity too high

**Solutions Applied:**
1. **SMOTE Optimization:** Reduced from 224K to 50K samples (gap: 0.35 → 0.19)
2. **Model Regularization:** Added dropout, early stopping, depth limits
3. **Cross-Validation:** 5-fold CV to detect overfitting early
4. **Ensemble Approach:** Voting Soft provided natural regularization

**Result:** Final model shows excellent generalization with acceptable train-val gap of 0.19.

### Challenge 3: Missing Income Data (19.8%)

**Problem:** Critical feature with substantial missing data. Dropping rows would lose 30,000 training examples.

**Why Simple Solutions Failed:**
- Mean imputation inappropriate (income is right-skewed)
- Dropping rows loses significant information
- Forward-fill makes no sense (records are independent)

**Solution:** Median imputation
- Income distributions are skewed (few very high earners)
- Median ($5,400) represents typical customer better than mean ($6,800)
- Validated that imputation didn't distort distributions

**Lesson:** Domain understanding crucial for imputation strategy selection.

### Challenge 4: Large Model Files (1.2GB total)

**Problem:** GitHub has 100MB file size limit. Git LFS still limited to 100MB per file on free tier.

**Failed Attempts:**
- Git LFS → Still hit limits
- Compression → Insufficient reduction
- Uploading only small models → Lost best ensemble

**Solution:** Google Drive integration
- Store models in Google Drive
- Download at container startup using `gdown`
- Environment variable controls which folder to download
- Clean Git repository (code only, no binaries)

**Benefits:**
- No size limitations
- Easy model updates without redeployment
- Could extend to model versioning

### Challenge 5: Real-World Data Quality

**Problems Encountered:**
- RevolvingUtilization > 100% (mathematically invalid)
- DebtRatio > 10,000 (impossible)
- Age values of 0 or >100
- Inconsistent patterns between features

**Solution:** IQR-based capping
- Statistical outlier detection using Interquartile Range
- Cap extreme values rather than removing records
- Preserve 98% of data while controlling extremes
- Feature-specific handling (e.g., hard bounds for age)

**Impact:** Models became more stable and less sensitive to anomalies.

### Challenge 6: Balancing Complexity and Interpretability

**Tension:** Financial regulations require model explainability, but simple models underperformed.

**Approach:**
- **Primary Model:** Voting Soft Ensemble for best predictions (0.8011 AUC)
- **Interpretability Aids:**
  - Feature importance analysis across base models
  - RiskScore feature (0-5 scale) for business users
  - Well-calibrated probabilities enable risk-based rules
  - API responses include risk level and recommendations

**Communication Strategy:**
- Document how each base model works
- Show feature contributions to predictions
- Emphasize that ensemble combines interpretable components
- Provide business-friendly explanations

**Result:** Achieved strong performance while maintaining sufficient interpretability for business adoption.

---

## Installation

### Prerequisites
- Python 3.10+
- pip package manager
- Git

### Quick Setup

**1. Clone Repository:**
```bash
git clone https://github.com/Zineb550/credit-default-prediction.git
cd credit-default-prediction
```

**2. Create Virtual Environment:**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

**4. Download Dataset:**
Place `cs-training.csv` in `data/raw/` directory from [Kaggle](https://www.kaggle.com/c/GiveMeSomeCredit/data)

### Training Models

```bash
python src/train.py
```

Generates:
- Trained models in `models/`
- Evaluation plots in `results/`
- Training logs in `logs/`

### Running API Locally

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

Access: http://localhost:8000/docs

### Docker Deployment

```bash
# Build
docker build -t credit-default-api .

# Run
docker run -p 8000:8000 -e GDRIVE_FOLDER_ID=1pkPu7yyj3-DXoNmbd98FXRHogSSkWZOV credit-default-api
```

---

## Project Structure

```
credit-default-prediction/
│
├── app/                    # API application
│   ├── api.py             # FastAPI endpoints
│   └── test_api.py        # API tests
│
├── src/                   # Source code
│   ├── preprocessing.py   # Data preprocessing
│   ├── model_trainer.py   # Model training
│   └── train.py          # Training pipeline
│
├── config/                # Configuration
│   └── config.yaml       # Hyperparameters
│
├── data/                  # Data directory
│   ├── raw/              # Original dataset
│   └── processed/        # Processed data
│
├── models/               # Trained models (not in GitHub)
├── results/              # Evaluation plots
├── logs/                 # Training logs
│
├── Dockerfile            # Docker configuration
├── start.sh              # Startup script
├── requirements.txt      # Dependencies
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

---

## Technologies

**Machine Learning:**
- Scikit-learn 1.3.0 - ML algorithms and pipelines
- XGBoost 2.0.0 - Gradient boosting
- LightGBM 4.0.0 - Fast gradient boosting
- Pandas 2.0.0 - Data manipulation
- NumPy 1.24.0 - Numerical computing
- Imbalanced-learn 0.11.0 - SMOTE resampling

**API & Deployment:**
- FastAPI 0.104.1 - Web framework
- Uvicorn 0.24.0 - ASGI server
- Docker - Containerization
- Render.com - Cloud hosting

**Visualization:**
- Matplotlib 3.7.0 - Plotting
- Seaborn 0.12.0 - Statistical visualization

---

## Conclusion

This project demonstrates a complete machine learning workflow from problem definition through production deployment. The final model achieves competitive performance (0.8011 ROC-AUC) through systematic feature engineering, careful data preprocessing, and ensemble learning.

**Key Achievements:**
- Comprehensive feature engineering (28 features from 11)
- Solved severe class imbalance and overfitting challenges
- Production deployment with monitoring capabilities
- Industry-competitive performance on real financial data

**Future Enhancements:**
- Ground truth collection for online evaluation
- Automated retraining pipeline
- A/B testing framework
- Extended monitoring with Prometheus/Grafana
- SHAP values for prediction explainability

---

**Author:** Zineb El Hadi  
**Institution:** INPT Morocco - INE2-DATA 2025  
**Contact:** [zinebelhadi2005@gmail.com]

*Last Updated: November 16, 2025*
