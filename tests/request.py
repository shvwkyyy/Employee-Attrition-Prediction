import requests

# Example payload matching your schema (update values as needed)
payload = {
    "Age": 30,
    "DailyRate": 800,
    "DistanceFromHome": 5,
    "Education": 3,
    "EnvironmentSatisfaction": 2,
    "HourlyRate": 60,
    "JobInvolvement": 3,
    "JobLevel": 2,
    "JobSatisfaction": 3,
    "MonthlyRate": 5000,
    "NumCompaniesWorked": 2,
    "PercentSalaryHike": 15,
    "PerformanceRating": 3,
    "RelationshipSatisfaction": 2,
    "StockOptionLevel": 1,
    "TotalWorkingYears": 10,
    "TrainingTimesLastYear": 2,
    "WorkLifeBalance": 3,
    "YearsAtCompany": 5,
    "YearsSinceLastPromotion": 1,
    "BusinessTravel": "Travel_Rarely",
    "Department": "Research & Development",
    "EducationField": "Life Sciences",
    "Gender": "Male",
    "JobRole": "Research Scientist",
    "MaritalStatus": "Single",
    "OverTime": "Yes"
}

response = requests.post("http://127.0.0.1:5000/predict", json=payload)
print("Status code:", response.status_code)
print("Response:", response.json())