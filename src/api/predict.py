import joblib
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError
from src.config import Config
from src.api.schemas import EmployeeData

logger = logging.getLogger(__name__)

class Predictor:
    """Handles model loading and predictions"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.pca = None
        self._load_models()

    def _load_models(self):
        """Load serialized models from disk"""
        try:
            self.model = joblib.load(Config.MODEL_DIR / 'model' / 'model.pkl')
            processor_dict = joblib.load(Config.MODEL_DIR / 'full_processor.joblib')
            self.preprocessor = processor_dict['preprocessor']  # Extract the preprocessor
            self.pca = processor_dict['pca']
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
            input_df = pd.DataFrame([employee.model_dump()])
            
            # Preprocess and predict
            processed_data = self.preprocessor.transform(input_df)
            processed_data = self.pca.transform(processed_data)
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
        if probability > 0.6:
            return "high"
        elif probability > 0.3:
            return "medium"
        return "low"

# Singleton instance
predictor = Predictor()