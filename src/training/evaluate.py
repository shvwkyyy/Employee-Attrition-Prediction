import mlflow
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import logging
from typing import Tuple, Dict
from config import Config
from data_processing.preprocess import load_and_preprocess_data

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation and metric tracking"""
    
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Run full evaluation suite"""
        try:
            # Generate predictions
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
            }
            
            # Generate classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            
            # Log to MLflow
            if Config.TRACKING_ENABLED:
                with mlflow.start_run():
                    mlflow.log_metrics(metrics)
                    mlflow.log_dict(report, "classification_report.json")
            
            logger.info(f"Evaluation completed. ROC AUC: {metrics['roc_auc']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            raise

def run_evaluation() -> Tuple[Dict[str, float], pd.DataFrame]:
    """Full evaluation pipeline"""
    # Load and preprocess data
    X, y, preprocessor = load_and_preprocess_data()
    
    # Split data (or use existing test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE,
        stratify=y
    )
    
    # Load production model
    model = joblib.load(Config.MODEL_DIR / 'production/model.pkl')
    
    # Evaluate
    evaluator = ModelEvaluator(model, preprocessor)
    metrics = evaluator.evaluate(X_test, y_test)
    
    return metrics, pd.DataFrame.from_dict(metrics, orient='index')

if __name__ == '__main__':
    metrics, metrics_df = run_evaluation()
    print("\nEvaluation Metrics:")
    print(metrics_df)