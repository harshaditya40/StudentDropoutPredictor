import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(
    page_title="Student Dropout Prediction System",
    layout="wide"
)

st.title("Student Dropout Prediction System")

st.write("""
This application predicts the likelihood of a student dropping out
based on academic, financial, and personal factors.
""")

st.info("Model Accuracy: 80.95%")

st.divider()

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=15.0,
        max_value=35.0,
        value=20.0,
        step=1.0
    )

    gender_text = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    gender = 0 if gender_text == "Female" else 1

    family_income = st.number_input(
        "Family Income (₹)",
        min_value=0,
        value=25000,
        step=1000
    )

    internet_text = st.selectbox(
        "Internet Access",
        ["No", "Yes"]
    )

    internet_access = 0 if internet_text == "No" else 1

    study_hours = st.number_input(
        "Study Hours Per Day",
        min_value=0.0,
        max_value=15.0,
        value=4.0,
        step=0.5
    )

    attendance = st.slider(
        "Attendance Rate (%)",
        0,
        100,
        75
    )

    assignment_delay = st.number_input(
        "Assignment Delay Days",
        min_value=0,
        max_value=60,
        value=5,
        step=1
    )

with col2:

    travel_time = st.number_input(
        "Travel Time (Minutes)",
        min_value=0,
        max_value=300,
        value=30,
        step=5
    )

    part_job_text = st.selectbox(
        "Part Time Job",
        ["No", "Yes"]
    )

    part_time_job = 0 if part_job_text == "No" else 1

    scholarship_text = st.selectbox(
        "Scholarship",
        ["No", "Yes"]
    )

    scholarship = 0 if scholarship_text == "No" else 1

    stress_index = st.slider(
        "Stress Index",
        0.0,
        10.0,
        5.0,
        0.1
    )

    gpa = st.number_input(
        "GPA",
        min_value=0.0,
        max_value=10.0,
        value=6.5,
        step=0.1
    )

    semester_gpa = st.number_input(
        "Semester GPA",
        min_value=0.0,
        max_value=10.0,
        value=6.5,
        step=0.1
    )

    cgpa = st.number_input(
        "CGPA",
        min_value=0.0,
        max_value=10.0,
        value=6.5,
        step=0.1
    )

    semester_text = st.selectbox(
        "Semester",
        ["1st", "2nd", "3rd", "4th"]
    )

    semester_map = {
        "1st": 0,
        "2nd": 1,
        "3rd": 2,
        "4th": 3
    }

    semester = semester_map[semester_text]

department_text = st.selectbox(
    "Department",
    [
        "Computer Science",
        "Information Technology",
        "Electronics",
        "Mechanical",
        "Civil"
    ]
)

department_map = {
    "Computer Science": 0,
    "Information Technology": 1,
    "Electronics": 2,
    "Mechanical": 3,
    "Civil": 4
}

department = department_map[department_text]

parent_text = st.selectbox(
    "Parental Education",
    [
        "No Formal Education",
        "Primary School",
        "High School",
        "Graduate",
        "Post Graduate"
    ]
)

parent_map = {
    "No Formal Education": 0,
    "Primary School": 1,
    "High School": 2,
    "Graduate": 3,
    "Post Graduate": 4
}

parental_education = parent_map[parent_text]

st.divider()

if st.button("Predict Risk", use_container_width=True):

    student_id = 1

    data = np.array([[
        student_id,
        age,
        gender,
        family_income,
        internet_access,
        study_hours,
        attendance,
        assignment_delay,
        travel_time,
        part_time_job,
        scholarship,
        stress_index,
        gpa,
        semester_gpa,
        cgpa,
        semester,
        department,
        parental_education
    ]])

    data = scaler.transform(data)

    prediction = model.predict(data)[0]
    if prediction == 1:
        st.error("Student is likely to drop out.")
    else:
        st.success("Student is likely to continue studies.")

    try:
        probability = model.predict_proba(data)[0][1] * 100
    except:
        probability = 50

    st.subheader("Prediction Result")

    if probability < 35:
        st.success(f"Low Risk ({probability:.2f}%)")

    elif probability < 70:
        st.warning(f"Medium Risk ({probability:.2f}%)")

    else:
        st.error(f"High Risk ({probability:.2f}%)")

    st.progress(int(probability))

    st.subheader("Recommendations")

    recommendations = []

    if attendance < 75:
        recommendations.append("- Improve attendance.")

    if study_hours < 3:
        recommendations.append("- Increase daily study hours.")

    if assignment_delay > 10:
        recommendations.append("- Submit assignments on time.")

    if stress_index > 7:
        recommendations.append("- Focus on stress management.")

    if gpa < 6:
        recommendations.append("- Seek academic support.")

    if len(recommendations) == 0:
        recommendations.append("- Current performance indicators are satisfactory.")

    for rec in recommendations:
        st.write(rec)

st.divider()
st.caption("Student Dropout Prediction System | Machine Learning Mini Project")
st.sidebar.title("Project Information")

st.sidebar.write("""
Model Used: Logistic Regression

Accuracy: 80.95%

Dataset: Student Dropout Dataset

Developer: Harshaditya Buri
""")