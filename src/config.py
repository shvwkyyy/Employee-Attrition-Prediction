import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

class Config:
    # Data paths
    RAW_DATA_PATH = BASE_DIR / 'data' / 'raw' / 'attrition_data.csv'
    PROCESSED_DATA_DIR = BASE_DIR / 'data' / 'processed'
    MODEL_DIR = BASE_DIR / 'models' / 'production'
    
    # MLflow settings
    MLFLOW_TRACKING_URI = "file://" + str(BASE_DIR / 'mlruns')
    MLFLOW_EXPERIMENT_NAME = "employee_attrition"
    
    # Model training params
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    # Features
    NUMERIC_FEATURES = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeNumber', 
        'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 'JobLevel',
        'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked',
        'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
        'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
        'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
        'YearsSinceLastPromotion', 'YearsWithCurrManager'
    ]
    
    CATEGORICAL_FEATURES = [
        'BusinessTravel', 'Department', 'EducationField', 'Gender',
        'JobRole', 'MaritalStatus', 'OverTime'
    ]
    
    TARGET = 'Attrition'
    
    @classmethod
    def ensure_directories_exist(cls):
        """Create necessary directories if they don't exist"""
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)