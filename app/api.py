"""
FastAPI Deployment for Credit Default Prediction
RESTful API for model predictions with monitoring
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import joblib
import pandas as pd
import numpy as np
import os
from datetime import datetime
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Credit Default Prediction API",
    description="API for predicting credit default risk",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load model and preprocessor at startup
model_path = os.path.join(project_root, "models", "best_model.pkl")
preprocessor_path = os.path.join(project_root, "models", "preprocessor.pkl")

try:
    model = joblib.load(model_path)
    preprocessor_state = joblib.load(preprocessor_path)
    logger.info("✓ Model and preprocessor loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None
    preprocessor_state = None

# Monitoring data store (in production, use a database)
predictions_log = []


# Pydantic models for API
class CustomerData(BaseModel):
    """Input data for a single customer"""
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., description="Credit utilization ratio", alias="RevolvingUtilizationOfUnsecuredLines")
    age: int = Field(..., description="Customer age")
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(0, description="Number of times 30-59 days past due", alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(..., description="Monthly debt payments / gross income")
    MonthlyIncome: float = Field(..., description="Monthly income")
    NumberOfOpenCreditLinesAndLoans: int = Field(..., description="Number of open credit lines")
    NumberOfTimes90DaysLate: int = Field(0, description="Number of times 90+ days late")
    NumberRealEstateLoansOrLines: int = Field(0, description="Number of real estate loans")
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(0, description="Number of times 60-89 days past due", alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: int = Field(0, description="Number of dependents")

    class Config:
        populate_by_name = True  # Allow both alias and field name
        schema_extra = {
            "example": {
                "RevolvingUtilizationOfUnsecuredLines": 0.5,
                "age": 45,
                "NumberOfTime30-59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.3,
                "MonthlyIncome": 5000,
                "NumberOfOpenCreditLinesAndLoans": 8,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60-89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2
            }
        }


class PredictionResponse(BaseModel):
    """Response with prediction results"""
    prediction: int
    probability: float
    risk_level: str
    recommendation: str
    timestamp: str


class BatchCustomerData(BaseModel):
    """Batch prediction input"""
    customers: List[CustomerData]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    timestamp: str


@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Credit Default Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "stats": "/stats",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Check API health status"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat()
    }


def preprocess_input(data: Dict) -> pd.DataFrame:
    """Preprocess input data using saved preprocessor"""
    try:
        # Convert field names from API format to match training data
        field_mapping = {
            'NumberOfTime30_59DaysPastDueNotWorse': 'NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTime60_89DaysPastDueNotWorse': 'NumberOfTime60-89DaysPastDueNotWorse'
        }
        
        # Apply mapping
        converted_data = {}
        for key, value in data.items():
            converted_data[field_mapping.get(key, key)] = value
        
        # Original 10 features from CSV
        base_features = [
            'RevolvingUtilizationOfUnsecuredLines',
            'age', 
            'NumberOfTime30-59DaysPastDueNotWorse',
            'DebtRatio',
            'MonthlyIncome',
            'NumberOfOpenCreditLinesAndLoans',
            'NumberOfTimes90DaysLate',
            'NumberRealEstateLoansOrLines',
            'NumberOfTime60-89DaysPastDueNotWorse',
            'NumberOfDependents'
        ]
        
        # Create DataFrame with base features
        df = pd.DataFrame([converted_data])
        
        # Ensure all base features are present
        for feature in base_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # Create engineered features (same as in preprocessing)
        df['TotalPastDue'] = (df['NumberOfTime30-59DaysPastDueNotWorse'] + 
                               df['NumberOfTime60-89DaysPastDueNotWorse'] + 
                               df['NumberOfTimes90DaysLate'])
        
        df['HasPastDue'] = (df['TotalPastDue'] > 0).astype(int)
        
        df['DebtToIncomeRatio'] = df['DebtRatio'] / (df['MonthlyIncome'] + 1)
        
        df['UtilizationBin'] = pd.cut(df['RevolvingUtilizationOfUnsecuredLines'], 
                                       bins=[0, 0.3, 0.6, 1.0, float('inf')], 
                                       labels=[0, 1, 2, 3])
        df['UtilizationBin'] = df['UtilizationBin'].astype(int)
        
        df['AgeBin'] = pd.cut(df['age'], 
                               bins=[0, 30, 50, 70, float('inf')], 
                               labels=[0, 1, 2, 3])
        df['AgeBin'] = df['AgeBin'].astype(int)
        
        df['CreditLinesPerDependent'] = df['NumberOfOpenCreditLinesAndLoans'] / (df['NumberOfDependents'] + 1)
        
        df['HighUtilization'] = (df['RevolvingUtilizationOfUnsecuredLines'] > 0.8).astype(int)
        df['YoungAge'] = (df['age'] < 25).astype(int)
        df['HighDebt'] = (df['DebtRatio'] > 0.5).astype(int)
        df['LowIncome'] = (df['MonthlyIncome'] < 3000).astype(int)
        df['ManyCreditLines'] = (df['NumberOfOpenCreditLinesAndLoans'] > 10).astype(int)
        df['HasRealEstate'] = (df['NumberRealEstateLoansOrLines'] > 0).astype(int)
        
        df['RiskScore'] = (df['HighUtilization'] + df['HighDebt'] + 
                           df['HasPastDue'] + df['LowIncome'] + df['ManyCreditLines'])
        
        df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)
        df['UtilizationDebtInteraction'] = df['RevolvingUtilizationOfUnsecuredLines'] * df['DebtRatio']
        df['AgeIncomeInteraction'] = df['age'] * df['MonthlyIncome']
        df['CreditUtilizationRatio'] = df['NumberOfOpenCreditLinesAndLoans'] * df['RevolvingUtilizationOfUnsecuredLines']
        
        df['TotalDelinquencies'] = df['TotalPastDue']
        
        # Select features in the correct order (28 features total)
        feature_names = preprocessor_state['feature_names']
        
        # Ensure we have all required features
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        # Select only the features used in training
        X = df[feature_names]
        
        # Apply scaling
        X_scaled = pd.DataFrame(
            preprocessor_state['scaler'].transform(X),
            columns=feature_names
        )
        
        return X_scaled
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise Exception(f"Preprocessing failed: {str(e)}")


def classify_risk(probability: float) -> tuple:
    """Classify risk level and provide recommendation"""
    if probability < 0.3:
        risk_level = "Low Risk"
        recommendation = "APPROVE - Low default risk. Proceed with standard terms."
    elif probability < 0.6:
        risk_level = "Medium Risk"
        recommendation = "REVIEW - Moderate default risk. Manual review recommended. Consider higher interest rate or additional collateral."
    else:
        risk_level = "High Risk"
        recommendation = "REJECT - High default risk. Loan approval not recommended."
    
    return risk_level, recommendation


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(customer: CustomerData):
    """
    Predict default probability for a single customer
    
    Returns:
    - prediction: 0 (No Default) or 1 (Default)
    - probability: Probability of default (0-1)
    - risk_level: Low/Medium/High Risk
    - recommendation: Business recommendation
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert to dict
        customer_dict = customer.dict()
        
        # Preprocess
        X = preprocess_input(customer_dict)
        
        # Predict
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])
        
        # Classify risk
        risk_level, recommendation = classify_risk(probability)
        
        # Log prediction for monitoring
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input": customer_dict,
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level
        }
        predictions_log.append(log_entry)
        
        # Keep only last 1000 predictions in memory
        if len(predictions_log) > 1000:
            predictions_log.pop(0)
        
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "recommendation": recommendation,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch-predict", tags=["Predictions"])
async def batch_predict(batch: BatchCustomerData):
    """
    Predict default probability for multiple customers
    
    Returns a list of predictions for each customer
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        results = []
        
        for customer in batch.customers:
            # Convert to dict
            customer_dict = customer.dict()
            
            # Preprocess
            X = preprocess_input(customer_dict)
            
            # Predict
            prediction = int(model.predict(X)[0])
            probability = float(model.predict_proba(X)[0][1])
            
            # Classify risk
            risk_level, recommendation = classify_risk(probability)
            
            results.append({
                "prediction": prediction,
                "probability": round(probability, 4),
                "risk_level": risk_level,
                "recommendation": recommendation
            })
        
        return {
            "total_customers": len(results),
            "predictions": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/stats", tags=["Monitoring"])
async def get_stats():
    """
    Get prediction statistics for monitoring
    
    Returns statistics about recent predictions
    """
    if len(predictions_log) == 0:
        return {
            "message": "No predictions yet",
            "total_predictions": 0
        }
    
    total = len(predictions_log)
    predictions = [p['prediction'] for p in predictions_log]
    probabilities = [p['probability'] for p in predictions_log]
    
    default_count = sum(predictions)
    default_rate = default_count / total if total > 0 else 0
    
    return {
        "total_predictions": total,
        "default_predictions": default_count,
        "default_rate": round(default_rate, 4),
        "average_probability": round(np.mean(probabilities), 4),
        "min_probability": round(np.min(probabilities), 4),
        "max_probability": round(np.max(probabilities), 4),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/monitoring/drift", tags=["Monitoring"])
async def check_drift():
    """
    Check for data drift in recent predictions
    
    Compares recent prediction distribution to expected distribution
    """
    if len(predictions_log) < 100:
        return {
            "message": "Need at least 100 predictions for drift detection",
            "current_predictions": len(predictions_log)
        }
    
    # Get recent predictions
    recent_probs = [p['probability'] for p in predictions_log[-100:]]
    
    # Expected default rate from training (6.7%)
    expected_default_rate = 0.067
    
    # Recent default rate
    recent_default_rate = np.mean([p['prediction'] for p in predictions_log[-100:]])
    
    # Calculate drift
    drift = abs(recent_default_rate - expected_default_rate)
    
    # Alert if drift > 5%
    alert = drift > 0.05
    
    return {
        "expected_default_rate": round(expected_default_rate, 4),
        "recent_default_rate": round(recent_default_rate, 4),
        "drift": round(drift, 4),
        "alert": alert,
        "status": "ALERT: Significant drift detected!" if alert else "Normal",
        "sample_size": len(predictions_log[-100:]),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/model-info", tags=["General"])
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": type(model).__name__,
        "features": preprocessor_state['feature_names'] if preprocessor_state else [],
        "n_features": len(preprocessor_state['feature_names']) if preprocessor_state else 0,
        "model_path": model_path,
        "loaded_at": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)