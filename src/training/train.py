import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, classification_report
)
import joblib
from pathlib import Path
from ..config import Config
from ..data_processing.preprocess import load_data, preprocess_and_save_data

def train_model():
    """Train and evaluate model with MLflow tracking"""
    Config.ensure_directories_exist()
    
    # Set up MLflow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
    
    # Load and preprocess data
    df = load_data()
    X, y, preprocessor = preprocess_and_save_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE,
        stratify=y
    )
    
    with mlflow.start_run():
        mlflow.log_params(Config.MODEL_PARAMS)
        mlflow.log_param("model_class", Config.MODEL_CLASS)
        
        model = eval(Config.MODEL_CLASS)(**Config.MODEL_PARAMS)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba)
        }
        
        # Log parameters and metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        
        # Log artifacts
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(Config.RAW_DATA_PATH, "data")
        
        # Save model to production
        mlflow.sklearn.save_model(model, str(Config.MODEL_DIR / 'model'))
        
        print(f"Model trained and saved. ROC AUC: {metrics['roc_auc']:.4f}")
        
    return model, metrics