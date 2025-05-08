from pydantic import BaseModel, Field
from typing import Literal

class EmployeeData(BaseModel):
    """Employee data schema with comprehensive validation"""
    # Numeric fields with range validation
    Age: int = Field(..., ge=18, le=70)
    DailyRate: int = Field(..., ge=0, le=3000)
    DistanceFromHome: int = Field(..., ge=0, le=30)
    Education: int = Field(..., ge=1, le=5)
    EnvironmentSatisfaction: int = Field(..., ge=1, le=4)
    HourlyRate: int = Field(..., ge=0, le=200)
    JobInvolvement: int = Field(..., ge=1, le=4)
    JobLevel: int = Field(..., ge=1, le=5)
    JobSatisfaction: int = Field(..., ge=1, le=4)
    MonthlyRate: int = Field(..., ge=0)
    NumCompaniesWorked: int = Field(..., ge=0, le=10)
    PercentSalaryHike: int = Field(..., ge=0, le=100)
    PerformanceRating: int = Field(..., ge=1, le=4)
    RelationshipSatisfaction: int = Field(..., ge=1, le=4)
    StockOptionLevel: int = Field(..., ge=0, le=3)
    TotalWorkingYears: int = Field(..., ge=0, le=50)
    TrainingTimesLastYear: int = Field(..., ge=0, le=10)
    WorkLifeBalance: int = Field(..., ge=1, le=4)
    YearsAtCompany: int = Field(..., ge=0, le=40)
    YearsSinceLastPromotion: int = Field(..., ge=0, le=15)

    # Categorical fields
    BusinessTravel: Literal['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']
    Department: Literal['Sales', 'Research & Development', 'Human Resources']
    EducationField: Literal[
        'Life Sciences', 'Medical', 'Marketing', 
        'Technical Degree', 'Other', 'Human Resources'
    ]
    Gender: Literal['Male', 'Female']
    JobRole: Literal[
        'Sales Executive', 'Research Scientist', 'Laboratory Technician', 
        'Manufacturing Director', 'Healthcare Representative', 
        'Manager', 'Sales Representative', 'Research Director', 
        'Human Resources'
    ]
    MaritalStatus: Literal['Single', 'Married', 'Divorced']
    OverTime: Literal['Yes', 'No']
    