import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score,
    classification_report
)
import logging
from typing import Dict
from src.config import Config

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles model evaluation and metric tracking"""
    
    @staticmethod
    def evaluate(y_true: pd.Series, y_pred: pd.Series, y_proba: pd.Series ) -> Dict[str, float]:
        """Run full evaluation suite"""
        try:
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred),
                'f1': f1_score(y_true, y_pred),
                'roc_auc':roc_auc_score(y_true, y_proba)
            }
            
            report = classification_report(y_true, y_pred, output_dict=True)
            
            if Config.TRACKING_ENABLED:
                mlflow.log_metrics(metrics)
                mlflow.log_dict(report, "classification_report.json")
            
            logger.info(f"Evaluation completed. Accuracy: {metrics['accuracy']:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {str(e)}", exc_info=True)
            raise