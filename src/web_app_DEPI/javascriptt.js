document.getElementById('predictButton').addEventListener('click', function() {
    let form = document.getElementById('predictionForm');
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

    // Parse numeric values
    let age = parseFloat(data.age);
    let totalWorkingYears = parseFloat(data.totalWorkingYears);
    let yearsAtCompany = parseFloat(data.yearsAtCompany);
    let yearsSinceLastPromotion = parseFloat(data.yearsSinceLastPromotion);
    let distanceFromHome = parseFloat(data.distanceFromHome);
    let numCompaniesWorked = parseFloat(data.numCompaniesWorked);
    let percentSalaryHike = parseFloat(data.percentSalaryHike);
    let dailyRate = parseFloat(data.dailyRate);
    let hourlyRate = parseFloat(data.hourlyRate);
    let monthlyRate = parseFloat(data.monthlyRate);

    // Validate numeric constraints
    if (isNaN(age) || age < 18 || age > 65 || !Number.isInteger(age)) {
        errors.push('Age must be an integer between 18 and 65');
    }
    if (isNaN(totalWorkingYears) || totalWorkingYears < 0 || totalWorkingYears > age-18 || !Number.isInteger(totalWorkingYears)) {
        errors.push('Total Working Years must be an integer between 0 and' + (age-18));
    }
    if (isNaN(yearsAtCompany) || yearsAtCompany < 0 || yearsAtCompany > totalWorkingYears || !Number.isInteger(yearsAtCompany)) {
        errors.push('Years at Company must be an integer between 0 and Total Working Years');
    }
    if (isNaN(yearsSinceLastPromotion) || yearsSinceLastPromotion < 0 || yearsSinceLastPromotion > yearsAtCompany || !Number.isInteger(yearsSinceLastPromotion)) {
        errors.push('Years Since Last Promotion must be an integer between 0 and Years at Company');
    }
    if (isNaN(distanceFromHome) || distanceFromHome < 0 || !Number.isInteger(distanceFromHome)) {
        errors.push('Distance from Home must be a non-negative integer');
    }
    if (isNaN(numCompaniesWorked) || numCompaniesWorked < 0 || !Number.isInteger(numCompaniesWorked)) {
        errors.push('Number of Companies Worked must be a non-negative integer');
    }
    if (isNaN(percentSalaryHike) || percentSalaryHike < 0 ) {
        errors.push('Percent Salary Hike must be between 0 and 100');
    }
    if (isNaN(dailyRate) || dailyRate <= 0) {
        errors.push('Daily Rate must be greater than 0');
    }
    if (isNaN(hourlyRate) || hourlyRate <= 0) {
        errors.push('Hourly Rate must be greater than 0');
    }
    if (isNaN(monthlyRate) || monthlyRate <= 0) {
        errors.push('Monthly Rate must be greater than 0');
    }

    if (errors.length > 0) {
        alert(errors.join('\n'));
        return;
    }

    // Proceed with prediction (placeholder)
    document.getElementById('result').innerText = 'Data collected: ' + JSON.stringify(data);
    // Replace with API call for actual prediction
});
