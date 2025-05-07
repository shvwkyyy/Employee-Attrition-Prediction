from pydantic import BaseModel, Field, field_validator, conint, constr
from typing import Literal, Optional

class EmployeeData(BaseModel):
    """Enhanced schema with built-in validation"""
    # Numeric fields with range validation
    from pydantic import Field  # Add this import at the top if not already present

    age: int = Field(..., ge=18, le=70)
    daily_rate: int = Field(..., ge=0, le=3000)
    distance_from_home: int = Field(..., ge=0, le=30)
    education: int = Field(..., ge=1, le=5)
    environment_satisfaction: int = Field(..., ge=1, le=4)
    monthly_income: int = Field(..., ge=0)
    
    # Categorical fields with enum validation
    business_travel: Literal['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']
    department: Literal['Sales', 'Research & Development', 'Human Resources']
    education_field: Literal[
        'Life Sciences', 'Medical', 'Marketing', 
        'Technical Degree', 'Other', 'Human Resources'
    ]
    gender: Literal['Male', 'Female']
    marital_status: Literal['Single', 'Married', 'Divorced']
    over_time: Literal['Yes', 'No']
    
    # Optional fields (if needed)
    employee_number: Optional[int] = None
    
    # Custom validation examples
    @field_validator('monthly_income')
    def validate_income(cls, v):
        if v > 1000000:  # Adjust threshold as needed
            raise ValueError('Unrealistically high monthly income')
        return v

    @field_validator('education_field')
    def validate_education_job_alignment(cls, v, values):
        if 'job_role' in values and v == 'Human Resources' and values['job_role'] != 'HR':
            raise ValueError('HR education field requires HR job role')
        return v