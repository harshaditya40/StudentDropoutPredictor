# Student Dropout Prediction System

## Overview

The Student Dropout Prediction System is a Machine Learning web application that predicts the likelihood of a student dropping out based on academic, financial, and personal factors.

The project uses a Logistic Regression model trained on a student dataset and is deployed using Streamlit Cloud.

## Features

- Predicts student dropout risk
- User-friendly web interface
- Real-time prediction
- Displays risk percentage
- Provides recommendations based on student inputs
- Fully deployed online using Streamlit

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Joblib

## Dataset Features

The model uses the following inputs:

- Age
- Gender
- Family Income
- Internet Access
- Study Hours Per Day
- Attendance Rate
- Assignment Delay Days
- Travel Time
- Part-Time Job Status
- Scholarship Status
- Stress Index
- GPA
- Semester GPA
- CGPA
- Semester
- Department
- Parental Education

## Machine Learning Model

- Algorithm: Logistic Regression
- Preprocessing: StandardScaler
- Model Storage: Joblib
- Accuracy: 80.95%

## Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Feature Selection
4. Model Training
5. Model Evaluation
6. Deployment using Streamlit

## Installation

Clone the repository:

```bash
git clone https://github.com/harshaditya40/StudentDropoutPredictor.git
```

Move to the project directory:

```bash
cd StudentDropoutPredictor
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run App.py
```

## Live Demo

Deployed using Streamlit Cloud.

https://studentdropoutpredictor-7.streamlit.app/

## Future Improvements

- Advanced ML models (Random Forest, XGBoost)
- Explainable AI (Feature Importance)
- Student Performance Dashboard
- Database Integration
- PDF Report Generation
- Early Warning System for Institutions

## Project Structure

```text
StudentDropoutPredictor/
│
├── App.py
├── best_model.pkl
├── scaler.pkl
├── student_dropout_dataset_v3.csv
├── requirements.txt
├── project.ipynb
└── README.md
```

## Author

Harshaditya Buri

B.Tech Student | Machine Learning Enthusiast

## License

This project is developed for educational and academic purposes.
