import joblib
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError
from config import Config
from api.schemas import EmployeeData

logger = logging.getLogger(__name__)

class Predictor:
    """Handles model loading and predictions"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self._load_models()

    def _load_models(self):
        """Load serialized models from disk"""
        try:
            self.model = joblib.load(Config.MODEL_DIR / 'model' / 'model.pkl')
            self.preprocessor = joblib.load(Config.MODEL_DIR / 'preprocessor.joblib')
            logger.info("Models loaded successfully")
        except Exception as e:
            logger.critical(f"Model loading failed: {str(e)}")
            raise RuntimeError("Failed to initialize prediction service")

    def validate_input(self, input_data: Dict[str, Any]) -> EmployeeData:
        """Validate input against Pydantic schema"""
        try:
            return EmployeeData(**input_data)
        except ValidationError as e:
            logger.warning(f"Input validation failed: {e.errors()}")
            raise ValueError({"errors": e.errors()})

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction on validated input"""
        try:
            # Validate and convert to DataFrame
            employee = self.validate_input(input_data)
            input_df = pd.DataFrame([employee.dict()])
            
            # Preprocess and predict
            processed_data = self.preprocessor.transform(input_df)
            prediction = self.model.predict(processed_data)
            probability = self.model.predict_proba(processed_data)[0, 1]
            
            return {
                'prediction': int(prediction[0]),
                'probability': float(probability),
                'confidence': self._get_confidence_level(probability)
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}", exc_info=True)
            raise

    def _get_confidence_level(self, probability: float) -> str:
        """Convert probability to confidence level"""
        if probability > 0.7:
            return "high"
        elif probability > 0.5:
            return "medium"
        return "low"

# Singleton instance
predictor = Predictor()