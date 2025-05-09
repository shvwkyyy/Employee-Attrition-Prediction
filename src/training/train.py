import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from src.config import Config
from src.data_processing.preprocess import load_data, preprocess_and_save_data
from src.training.evaluate import ModelEvaluator
# Add to your imports
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
import numpy as np

# Add to config.py (or ensure these exist)
class Config:
    API_PORT = 8000
    API_HOST = "0.0.0.0"
    MODEL_ARTIFACT_PATH = "mlruns/0/<RUN_ID>/artifacts/model"  # Replace <RUN_ID>

def train_model():
    """Train and evaluate model with cross-validation and MLflow tracking"""
    Config.ensure_dirs_exist()
    
    # Set up MLflow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
    
    # Load and preprocess data
    df = load_data()
    X, y, preprocessor = preprocess_and_save_data(df)
    
    with mlflow.start_run(run_name= 'Ensemble') as run:
        print(f"Run ID: {run.info.run_id}")
        # Log parameters in chunks
        mlflow.log_param("model_type", "StackingClassifier")
        
        # Log individual estimator parameters
        for i, (name, estimator) in enumerate(Config.MODEL_PARAMS['estimators']):
            mlflow.log_param(f"estimator_{i}_type", name)
            for param, value in estimator.get_params().items():
                mlflow.log_param(f"estimator_{i}_{param}", value)
        
        # Log final estimator parameters
        mlflow.log_param("final_estimator_type", 
                        type(Config.MODEL_PARAMS['final_estimator']).__name__)
        for param, value in Config.MODEL_PARAMS['final_estimator'].get_params().items():
            mlflow.log_param(f"final_estimator_{param}", value)
        
        # Initialize model and CV
        model = StackingClassifier(**Config.MODEL_PARAMS)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=Config.RANDOM_STATE)
        
        # Get cross-validated predictions
        y_pred = cross_val_predict(model, X, y, cv=cv, method='predict')
        y_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        
        # Evaluate using ModelEvaluator
        metrics = ModelEvaluator.evaluate(y, y_pred, y_proba)
        
        # Retrain on full data for production
        model.fit(X, y)
        
        # Log artifacts
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_dict({
            "cv_predictions": y_pred.tolist(),
            "cv_probabilities": y_proba.tolist(),
            "true_labels": y.tolist()
        }, "cross_val_results.json")
        
        print(f"Model trained. CV ROC AUC: {metrics['roc_auc']:.4f}")
        mlflow.sklearn.save_model(model, str(Config.MODEL_DIR / 'model'))
        
    return model, metrics

if __name__ == "__main__":
    model, metrics = train_model()
    print("\nTraining completed!")
    print("Cross-validated metrics:")
    for metric, value in metrics.items():
        if metric != 'confusion_matrix':  # Skip printing the matrix
            print(f"{metric:>10}: {value:.4f}")