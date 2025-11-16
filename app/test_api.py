"""
Test Script for Credit Default Prediction API
Tests all endpoints with sample data
"""

import requests
import json
from pprint import pprint

# API base URL
BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    pprint(response.json())


def test_single_prediction():
    """Test single customer prediction"""
    print("\n" + "="*60)
    print("Testing Single Prediction Endpoint")
    print("="*60)
    
    # Sample customer data - Low risk
    customer_low_risk = {
        "RevolvingUtilizationOfUnsecuredLines": 0.2,
        "age": 45,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.15,
        "MonthlyIncome": 8000,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 2
    }
    
    print("\n📊 Customer Profile (Low Risk):")
    pprint(customer_low_risk)
    
    response = requests.post(f"{BASE_URL}/predict", json=customer_low_risk)
    print(f"\nStatus Code: {response.status_code}")
    print("\n🎯 Prediction Result:")
    pprint(response.json())
    
    # Sample customer data - High risk
    print("\n" + "-"*60)
    
    customer_high_risk = {
        "RevolvingUtilizationOfUnsecuredLines": 0.95,
        "age": 28,
        "NumberOfTime30_59DaysPastDueNotWorse": 3,
        "DebtRatio": 0.85,
        "MonthlyIncome": 2500,
        "NumberOfOpenCreditLinesAndLoans": 12,
        "NumberOfTimes90DaysLate": 2,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60_89DaysPastDueNotWorse": 1,
        "NumberOfDependents": 3
    }
    
    print("\n📊 Customer Profile (High Risk):")
    pprint(customer_high_risk)
    
    response = requests.post(f"{BASE_URL}/predict", json=customer_high_risk)
    print(f"\nStatus Code: {response.status_code}")
    print("\n🎯 Prediction Result:")
    pprint(response.json())


def test_batch_prediction():
    """Test batch prediction"""
    print("\n" + "="*60)
    print("Testing Batch Prediction Endpoint")
    print("="*60)
    
    customers = {
        "customers": [
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.3,
                "age": 40,
                "NumberOfTime30_59DaysPastDueNotWorse": 0,
                "DebtRatio": 0.2,
                "MonthlyIncome": 6000,
                "NumberOfOpenCreditLinesAndLoads": 6,
                "NumberOfTimes90DaysLate": 0,
                "NumberRealEstateLoansOrLines": 1,
                "NumberOfTime60_89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 1
            },
            {
                "RevolvingUtilizationOfUnsecuredLines": 0.9,
                "age": 25,
                "NumberOfTime30_59DaysPastDueNotWorse": 2,
                "DebtRatio": 0.7,
                "MonthlyIncome": 3000,
                "NumberOfOpenCreditLinesAndLoans": 10,
                "NumberOfTimes90DaysLate": 1,
                "NumberRealEstateLoansOrLines": 0,
                "NumberOfTime60_89DaysPastDueNotWorse": 0,
                "NumberOfDependents": 2
            }
        ]
    }
    
    print(f"\n📊 Batch Size: {len(customers['customers'])} customers")
    
    response = requests.post(f"{BASE_URL}/batch-predict", json=customers)
    print(f"\nStatus Code: {response.status_code}")
    print("\n🎯 Batch Prediction Results:")
    pprint(response.json())


def test_stats():
    """Test statistics endpoint"""
    print("\n" + "="*60)
    print("Testing Statistics Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status Code: {response.status_code}")
    print("\n📊 Prediction Statistics:")
    pprint(response.json())


def test_drift():
    """Test drift detection endpoint"""
    print("\n" + "="*60)
    print("Testing Drift Detection Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/monitoring/drift")
    print(f"Status Code: {response.status_code}")
    print("\n📊 Drift Analysis:")
    pprint(response.json())


def test_model_info():
    """Test model info endpoint"""
    print("\n" + "="*60)
    print("Testing Model Info Endpoint")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/model-info")
    print(f"Status Code: {response.status_code}")
    print("\n📊 Model Information:")
    result = response.json()
    print(f"Model Type: {result.get('model_type')}")
    print(f"Number of Features: {result.get('n_features')}")
    print(f"Features: {result.get('features')[:5]}... (showing first 5)")


def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*60)
    print("🚀 CREDIT DEFAULT PREDICTION API - TESTS")
    print("="*60)
    
    try:
        test_health()
        test_single_prediction()
        test_batch_prediction()
        test_stats()
        test_drift()
        test_model_info()
        
        print("\n" + "="*60)
        print("✅ All tests completed successfully!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to API")
        print("Make sure the API is running:")
        print("  python app/api.py")
        print("  or")
        print("  uvicorn app.api:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")


if __name__ == "__main__":
    run_all_tests()
