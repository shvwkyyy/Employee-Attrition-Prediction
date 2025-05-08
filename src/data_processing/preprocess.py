import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from pathlib import Path
from imblearn.pipeline import Pipeline  
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
import logging
from src.config import Config


logger = logging.getLogger(__name__)

def load_data(file_path: str = None) -> pd.DataFrame:
    """Load and validate raw data"""
    file_path = file_path or Config.RAW_DATA_PATH
    df = pd.read_csv(file_path)
    return df

def drop_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop specified columns from DataFrame"""
    columns=['Over18','EmployeeCount','StandardHours','EmployeeNumber','MonthlyIncome','YearsInCurrentRole','YearsWithCurrManager']
    df.drop(columns=columns,inplace=True)
    return df

def create_preprocessor():
    """Create preprocessing pipeline"""
    categorical_transformer = Pipeline(steps=[
        ('label', LabelEncoder(handle_unknown='ignore'))
    ])
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, Config.CATEGORICAL_FEATURES),
            ('num', numeric_transformer, Config.NUMERIC_FEATURES)
        ])
    
    return preprocessor

def preprocess_and_save_data(df: pd.DataFrame, save: bool = True):
    """Preprocess data and optionally save the preprocessor"""
    drop_columns(df)
    preprocessor = create_preprocessor()
    
    X = df.drop(Config.TARGET, axis=1)
    y = df[Config.TARGET].map({'Yes': 1, 'No': 0})
    sharmot=Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA(n_components=23)),
    ])
    X = sharmot.fit_resample(X)


    
    X_processed = preprocessor.fit_transform(X)
    
    if save:
        joblib.dump(preprocessor, Config.MODEL_DIR / 'preprocessor.joblib')
        pd.DataFrame(X_processed).to_csv(
            Config.PROCESSED_DIR / 'processed_features.csv', index=False)
        y.to_csv(Config.PROCESSED_DIR / 'target.csv', index=False)
    
    return X_processed, y, preprocessor