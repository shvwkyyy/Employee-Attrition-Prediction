import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import logging
from src.config import Config

logger = logging.getLogger(__name__)
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict
import logging
from src.config import Config  # Absolute import

logger = logging.getLogger(__name__)
class DataValidator:
    """Validates raw data meets expected standards"""
    
    def __init__(self):
        self.cfg = Config()
        self.expected_dtypes = {
            'Age': 'int64',
            'Attrition': 'object',
            'BusinessTravel': 'object',
            'DailyRate': 'int64',
            'Department': 'object',
            # Add all expected columns with types
        }
        self.value_constraints = {
            'Age': (18, 70),
            'Education': (1, 5),  # Likert scale 1-5
            'EnvironmentSatisfaction': (1, 4),
            # Add value ranges for numeric features
        }
        self.category_requirements = {
            'Attrition': ['Yes', 'No'],
            'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
            # Add valid categories for categorical features
        }

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, Dict]:
        """Run complete validation suite"""
        validation_results = {
            'missing_values': self.check_missing_values(df),
            'dtype_checks': self.check_dtypes(df),
            'value_ranges': self.check_value_ranges(df),
            'category_checks': self.check_categories(df),
            'id_duplicates': self.check_duplicates(df, 'EmployeeNumber')
        }
        
        is_valid = all(validation_results.values())
        
        if not is_valid:
            logger.error("Data validation failed. Results: %s", validation_results)
            raise ValueError(f"Data validation failed: {validation_results}")
        
        logger.info("Data validation passed all checks")
        return is_valid, validation_results

    def check_missing_values(self, df: pd.DataFrame) -> bool:
        """Check for unexpected null values"""
        null_counts = df.isnull().sum()
        problematic_cols = null_counts[null_counts > 0]
        
        if not problematic_cols.empty:
            logger.warning("Missing values found:\n%s", problematic_cols)
            return False
        return True

    def check_dtypes(self, df: pd.DataFrame) -> bool:
        """Verify column data types match expectations"""
        type_violations = []
        for col, expected_type in self.expected_dtypes.items():
            if str(df[col].dtype) != expected_type:
                type_violations.append(f"{col}: {df[col].dtype} ≠ {expected_type}")
        
        if type_violations:
            logger.warning("Dtype violations:\n%s", "\n".join(type_violations))
            return False
        return True

    def check_value_ranges(self, df: pd.DataFrame) -> bool:
        """Validate numeric value ranges"""
        violations = []
        for col, (min_val, max_val) in self.value_constraints.items():
            if (df[col].min() < min_val) or (df[col].max() > max_val):
                violations.append(f"{col} outside {min_val}-{max_val} range")
        
        if violations:
            logger.warning("Value range violations:\n%s", "\n".join(violations))
            return False
        return True

    def check_categories(self, df: pd.DataFrame) -> bool:
        """Validate categorical values"""
        violations = []
        for col, valid_categories in self.category_requirements.items():
            invalid = set(df[col].unique()) - set(valid_categories)
            if invalid:
                violations.append(f"{col} contains invalid categories: {invalid}")
        
        if violations:
            logger.warning("Category violations:\n%s", "\n".join(violations))
            return False
        return True

    def check_duplicates(self, df: pd.DataFrame, id_col: str) -> bool:
        """Check for duplicate IDs"""
        if df[id_col].duplicated().any():
            logger.warning("Duplicate IDs found in column %s", id_col)
            return False
        return True
 
def validate_raw_data(file_path: Path = None) -> pd.DataFrame:
    """Public interface for data validation"""
    validator = DataValidator()
    file_path = file_path or Config.RAW_DATA_PATH
    df = pd.read_csv(file_path)
    
    is_valid, _ = validator.validate_data(df)
    if not is_valid:
        raise ValueError("Data validation failed - check logs for details")
    
    return df
