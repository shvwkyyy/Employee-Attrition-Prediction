def validate_input(input_data):
    """Validate input data for prediction"""
    errors = []
    
    # Required fields
    required_fields = [
        'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
        'Education', 'EducationField', 'EmployeeNumber', 'EnvironmentSatisfaction',
        'Gender', 'JobRole', 'MaritalStatus', 'MonthlyIncome', 'OverTime'
    ]
    
    for field in required_fields:
        if field not in input_data:
            errors.append(f"Missing required field: {field}")
    
    # Type validation
    numeric_fields = {
        'Age': (18, 70),
        'DailyRate': (0, 3000),
        'DistanceFromHome': (0, 30),
        'Education': (1, 5),
        'EmployeeNumber': (1, None),
        'EnvironmentSatisfaction': (1, 4),
        'MonthlyIncome': (0, None)
    }
    
    for field, (min_val, max_val) in numeric_fields.items():
        if field in input_data:
            try:
                value = int(input_data[field])
                if min_val is not None and value < min_val:
                    errors.append(f"{field} must be >= {min_val}")
                if max_val is not None and value > max_val:
                    errors.append(f"{field} must be <= {max_val}")
            except ValueError:
                errors.append(f"{field} must be a number")
    
    # Categorical validation
    valid_categories = {
        'BusinessTravel': ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently'],
        'Department': ['Sales', 'Research & Development', 'Human Resources'],
        'EducationField': [
            'Life Sciences', 'Medical', 'Marketing', 'Technical Degree',
            'Other', 'Human Resources'
        ],
        'Gender': ['Male', 'Female'],
        'MaritalStatus': ['Single', 'Married', 'Divorced'],
        'OverTime': ['Yes', 'No']
    }
    
    for field, valid_values in valid_categories.items():
        if field in input_data and input_data[field] not in valid_values:
            errors.append(f"{field} must be one of: {', '.join(valid_values)}")
    
    return errors