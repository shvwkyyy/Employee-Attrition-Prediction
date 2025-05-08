import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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
    return df.drop(columns=Config.DELETED_FEATURES)

def create_preprocessor():
    """Create preprocessing pipeline"""
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Use OneHotEncoder instead of LabelEncoder for categorical features
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
    try:
        # 1. Drop columns
        df = drop_columns(df)
        
        # 2. Separate features and target
        X = df.drop(Config.TARGET, axis=1)
        y = df[Config.TARGET].map({'Yes': 1, 'No': 0})
        
        # 3. Create and apply preprocessor
        preprocessor = create_preprocessor()
        X_processed = preprocessor.fit_transform(X)
        
        # 4. Apply SMOTE first (before PCA) to balance classes
        smote = SMOTE(random_state=Config.RANDOM_STATE)
        X_resampled, y_resampled = smote.fit_resample(X_processed, y)
        
        # 5. Apply PCA to the resampled data
        pca = PCA(n_components=23)
        X_final = pca.fit_transform(X_resampled)
        
        if save:
            # Save processors
            joblib.dump({
                'preprocessor': preprocessor,
                'pca': pca,
                'smote': smote
            }, Config.MODEL_DIR / 'full_processor.joblib')
            
            # Save processed data
            pd.DataFrame(X_final).to_csv(
                Config.PROCESSED_DIR / 'processed_features.csv', 
                index=False
            )
            pd.Series(y_resampled).to_csv(
                Config.PROCESSED_DIR / 'target.csv', 
                index=False
            )
        
        return X_final, y_resampled, {
            'preprocessor': preprocessor,
            'pca': pca,
            'smote': smote
        }
    
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise