let levels_map = {
    'low': 1,
    'medium': 2,
    'high': 3,
    'very high': 4,
};

let education_map = {
    'Below College': 1,
    'College': 2,
    'Bachelor': 3,
    'Master': 4,
    'Doctor': 5
};

let job_level_map = {
    'junior': 1,
    'mid_level': 2,
    'senior': 3,
    'manager': 4,
    'executive': 5,
};

document.getElementById('PredictButton').addEventListener('click', function () {
    let form = document.getElementById('PredictionForm');
    let inputs = form.querySelectorAll('input, select');
    let data = {};
    let errors = [];

    // Check required fields
    inputs.forEach(input => {
        if (input.required && !input.value) {
            errors.push(`${input.labels[0].textContent} is required`);
        } else {
            data[input.id] = input.value;
        }
    });

    if (errors.length > 0) {
        alert(errors.join('\n'));
        return;
    }

    // Apply mappings
    if (data.Education && education_map[data.Education]) {
        data.Education = education_map[data.Education];
    }
    if (data.JobLevel && job_level_map[data.JobLevel]) {
        data.JobLevel = job_level_map[data.JobLevel];
    }
    if (data.EnvironmentSatisfaction && levels_map[data.EnvironmentSatisfaction]) {
        data.EnvironmentSatisfaction = levels_map[data.EnvironmentSatisfaction];
    }
    if (data.JobInvolvement && levels_map[data.JobInvolvement]) {
        data.JobInvolvement = levels_map[data.JobInvolvement];
    }
    if (data.JobSatisfaction && levels_map[data.JobSatisfaction]) {
        data.JobSatisfaction = levels_map[data.JobSatisfaction];
    }
    if (data.RelationshipSatisfaction && levels_map[data.RelationshipSatisfaction]) {
        data.RelationshipSatisfaction = levels_map[data.RelationshipSatisfaction];
    }
    if (data.WorkLifeBalance && levels_map[data.WorkLifeBalance]) {
        data.WorkLifeBalance = levels_map[data.WorkLifeBalance];
    }
    if (data.PerformanceRating && levels_map[data.PerformanceRating]) {
        data.PerformanceRating = levels_map[data.PerformanceRating];
    }
    if (data.StockOptionLevel && levels_map[data.StockOptionLevel]) {
        data.StockOptionLevel = levels_map[data.StockOptionLevel];
    }

    // Parse numeric values
    let Age = parseFloat(data.Age);
    let TotalWorkingYears = parseFloat(data.TotalWorkingYears);
    let YearsAtCompany = parseFloat(data.YearsAtCompany);
    let YearsSinceLastPromotion = parseFloat(data.YearsSinceLastPromotion);
    let DistanceFromHome = parseFloat(data.DistanceFromHome);
    let NumCompaniesWorked = parseFloat(data.NumCompaniesWorked);
    let PercentSalaryHike = parseFloat(data.PercentSalaryHike);
    let DailyRate = parseFloat(data.DailyRate);
    let HourlyRate = parseFloat(data.HourlyRate);
    let MonthlyRate = parseFloat(data.MonthlyRate);

    // Validate numeric constraints
    if (isNaN(Age) || Age < 18 || Age > 65 || !Number.isInteger(Age)) {
        errors.push('Age must be an integer between 18 and 65');
    }
    if (isNaN(TotalWorkingYears) || TotalWorkingYears < 0 || TotalWorkingYears > Age - 18 || !Number.isInteger(TotalWorkingYears)) {
        errors.push('Total Working Years must be an integer between 0 and' + (Age - 18));
    }
    if (isNaN(YearsAtCompany) || YearsAtCompany < 0 || YearsAtCompany > TotalWorkingYears || !Number.isInteger(YearsAtCompany)) {
        errors.push('Years at Company must be an integer between 0 and Total Working Years');
    }
    if (isNaN(YearsSinceLastPromotion) || YearsSinceLastPromotion < 0 || YearsSinceLastPromotion > YearsAtCompany || !Number.isInteger(YearsSinceLastPromotion)) {
        errors.push('Years Since Last Promotion must be an integer between 0 and Years at Company');
    }
    if (isNaN(DistanceFromHome) || DistanceFromHome < 0 || !Number.isInteger(DistanceFromHome)) {
        errors.push('Distance from Home must be a non-negative integer');
    }
    if (isNaN(NumCompaniesWorked) || NumCompaniesWorked < 0 || !Number.isInteger(NumCompaniesWorked)) {
        errors.push('Number of Companies Worked must be a non-negative integer');
    }
    if (isNaN(PercentSalaryHike) || PercentSalaryHike < 0) {
        errors.push('Percent Salary Hike must be between 0 and 100');
    }
    if (isNaN(DailyRate) || DailyRate <= 0) {
        errors.push('Daily Rate must be greater than 0');
    }
    if (isNaN(HourlyRate) || HourlyRate <= 0) {
        errors.push('Hourly Rate must be greater than 0');
    }
    if (isNaN(MonthlyRate) || MonthlyRate <= 0) {
        errors.push('Monthly Rate must be greater than 0');
    }

    if (errors.length > 0) {
        alert(errors.join('\n'));
        return;
    }

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
        .then(response => response.json())
        .then(prediction => {
            console.log(prediction);
            document.getElementById('Result').innerText = prediction['prediction']?'The employee is likely to leave the company.':'The employee is likely to stay in the company.';
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while fetching the prediction. Please try again.');
        });
});