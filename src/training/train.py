import mlflow
import mlflow.sklearn
from src.config import Config
from src.data_processing.preprocess import load_data, preprocess_and_save_data
from src.training.evaluate import ModelEvaluator
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import  StratifiedKFold

def train_model():
    """Train and evaluate model with MLflow tracking"""
    Config.ensure_dirs_exist()
    
    # Set up MLflow
    mlflow.set_tracking_uri(Config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(Config.MLFLOW_EXPERIMENT_NAME)
    
    # Load and preprocess data
    df = load_data()
    X, y, preprocessor = preprocess_and_save_data(df)
    
    # Split data
    
    with mlflow.start_run() as run:
        print(f"Run ID: {run.info.run_id}")
        mlflow.log_params(Config.MODEL_PARAMS)
        
        model = StackingClassifier(**Config.MODEL_PARAMS)
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

        y_pred = cross_val_predict(model, X, y, cv=cv)        
        # Evaluate 
        evaluator = ModelEvaluator.evaluate(model, preprocessor)
        
        # Log artifacts
        mlflow.sklearn.log_model(model, "model")
        mlflow.log_artifact(Config.RAW_DATA_PATH, "data")
        
        # Save model to production
        mlflow.sklearn.save_model(model, str(Config.MODEL_DIR / 'model'))
        
    return model, metrics


if __name__ == "__main__":
    model, metrics = train_model()
    print("Training completed!")
    print(f"Model accuracy: {metrics['accuracy']:.2f}")