# Employee Attrition Prediction Model

## Overview
This project aims to predict employee attrition using machine learning techniques. The model helps organizations identify potential risks of employees leaving and enables proactive retention strategies.

## Features
- Data preprocessing and feature engineering
- Handling class imbalance using SMOTE and upsampling techniques
- Model training and evaluation with multiple machine learning algorithms
- Deployment using Docker

## Technologies Used
- Python
- Pandas, NumPy, Matplotlib, Seaborn (Data Analysis & Visualization)
- Scikit-learn (Machine Learning)
- Imbalanced-learn (SMOTE for handling class imbalance)
- Flask (API for model inference)
- Docker (Containerization)

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/employee-attrition.git
   cd employee-attrition
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Train the model:
   ```bash
   python train.py
   ```
2. Run the API server:
   ```bash
   python app.py
   ```
3. Use Docker for deployment:
   ```bash
   docker build -t employee-attrition .
   docker run -p 5000:5000 employee-attrition
   ```

## API Endpoints
- `POST /predict` - Predicts attrition for a given employee profile (expects JSON input).

## Dataset
The dataset used in this project includes various employee-related features such as age, job role, salary, and work-life balance. It is preprocessed before feeding into the model.

## Results
- Achieved high accuracy and precision in predicting employee attrition.
- Improved model performance using SMOTE and feature engineering.

## Future Enhancements
- Implement deep learning models for better accuracy.
- Develop a web-based dashboard for visualization.
- Integrate real-time data processing.


