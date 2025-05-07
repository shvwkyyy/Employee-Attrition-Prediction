import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
import logging
from src.config import Config
from src.data_processing.data_validation import DataValidator  # Updated import

logger = logging.getLogger(__name__)

def load_data(file_path: str = None) -> pd.DataFrame:
    """Load and validate raw data"""
    file_path = file_path or Config.RAW_DATA_PATH
    df = pd.read_csv(file_path)
    DataValidator().validate_data(df)  # Updated validation call
    return df

def create_preprocessor():
    """Create preprocessing pipeline"""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, Config.NUMERIC_FEATURES),
            ('cat', categorical_transformer, Config.CATEGORICAL_FEATURES)
        ])
    
    return preprocessor

def preprocess_and_save_data(df: pd.DataFrame, save: bool = True):
    """Preprocess data and optionally save the preprocessor"""
    preprocessor = create_preprocessor()
    
    X = df.drop(Config.TARGET, axis=1)
    y = df[Config.TARGET].map({'Yes': 1, 'No': 0})
    
    X_processed = preprocessor.fit_transform(X)
    
    if save:
        joblib.dump(preprocessor, Config.MODEL_DIR / 'preprocessor.joblib')
        pd.DataFrame(X_processed).to_csv(
            Config.PROCESSED_DIR / 'processed_features.csv', index=False)
        y.to_csv(Config.PROCESSED_DIR / 'target.csv', index=False)
    
    return X_processed, y, preprocessor