from pathlib import Path
import os
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    RAW_DATA_PATH = BASE_DIR / "data/raw/Employee-Attrition.csv"
    PROCESSED_DIR = BASE_DIR / "data/processed"
    MODEL_DIR = BASE_DIR / "models/production"
    
    # MLflow Settings
    TRACKING_ENABLED = True
    MLFLOW_TRACKING_URI = "file://" + str(BASE_DIR / "mlruns")
    MLFLOW_EXPERIMENT_NAME = "employee_attrition"
    MLFLOW_REGISTERED_MODEL_NAME = "stacking-classifier"  # ✅ اسم الموديل داخل Model Registry
    
    # Model Parameters
    estimators = [
        ('lr', LogisticRegression(C=1, penalty='l1', class_weight='balanced', solver='liblinear', random_state=42)),
        ('xg', XGBClassifier(colsample_bytree=1.0, learning_rate=0.01, max_depth=3, n_estimators=100, subsample=0.8, random_state=42)),
        ('ds', DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=5, min_samples_split=5, random_state=42))
    ]
    
    final_estimator = LogisticRegression(C=1, penalty='l1', class_weight='balanced', solver='liblinear', random_state=42)
    
    MODEL_PARAMS = {
        'estimators': estimators,
        'final_estimator': final_estimator
    }
    
    # Features
    NUMERIC_FEATURES = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 
        'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 
        'JobLevel', 'JobSatisfaction', 'MonthlyRate', 
        'NumCompaniesWorked', 'PercentSalaryHike', 'PerformanceRating', 
        'RelationshipSatisfaction', 'StockOptionLevel',
        'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
        'YearsAtCompany', 'YearsSinceLastPromotion'
    ]
    
    DELETED_FEATURES = [
        'Over18', 'EmployeeCount', 'StandardHours',
        'EmployeeNumber', 'MonthlyIncome',
        'YearsInCurrentRole', 'YearsWithCurrManager'
    ]
    
    CATEGORICAL_FEATURES = [
        'BusinessTravel', 'Department', 'EducationField', 'Gender', 
        'JobRole', 'MaritalStatus', 'OverTime'
    ]
    
    TARGET = 'Attrition'
    
    # Validation
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    @classmethod
    def ensure_dirs_exist(cls):
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)
