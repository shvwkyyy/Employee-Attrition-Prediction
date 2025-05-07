from pathlib import Path
import os

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent.parent
    RAW_DATA_PATH = BASE_DIR / "data/raw/attrition_data.csv"
    PROCESSED_DIR = BASE_DIR / "data/processed"
    MODEL_DIR = BASE_DIR / "models/production"
    
    # MLflow Settings
    MLFLOW_TRACKING_URI = "file://" + str(BASE_DIR / "mlruns")
    MLFLOW_EXPERIMENT_NAME = "employee_attrition"
    
    # Model Parameters
    MODEL_CLASS = "sklearn.ensemble.RandomForestClassifier"
    MODEL_PARAMS = {
        "n_estimators": 100,
        "max_depth": 8,
        "random_state": 42,
        "class_weight": "balanced"
    }
    
    # Features
    NUMERIC_FEATURES = [
        'Age', 'DailyRate', 'DistanceFromHome', 'Education', 
        'EnvironmentSatisfaction', 'MonthlyIncome'
    ]
    
    CATEGORICAL_FEATURES = [
        'BusinessTravel', 'Department', 'Gender', 
        'JobRole', 'OverTime'
    ]
    
    TARGET = 'Attrition'
    
    # Validation
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    @classmethod
    def ensure_dirs_exist(cls):
        os.makedirs(cls.PROCESSED_DIR, exist_ok=True)
        os.makedirs(cls.MODEL_DIR, exist_ok=True)