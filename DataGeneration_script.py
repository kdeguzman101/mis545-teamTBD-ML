# This is the script we used to generate our dataset.
import pandas as pd
import numpy as np

np.random.seed(42)

num_records = 10000

sat_scores = np.random.normal(loc=1028, scale=221, size=num_records).clip(800, 1600)
act_scores = (sat_scores / 1600 * 36).astype(int).clip(15, 36)
gpa = (sat_scores / 1600 * 4.0 + np.random.normal(0, 0.3, num_records)).clip(2.0, 4.0)
retention_rate = (gpa / 4.0 + np.random.normal(0, 0.1, num_records)).clip(0.5, 1.0)
graduation_rate = (retention_rate + np.random.normal(0, 0.05, num_records)).clip(0.4, 1.0)
starting_age = np.random.normal(loc=18.5, scale=0.5, size=num_records).clip(17, 20)
graduation_age = (starting_age + 4 + (1 - retention_rate) * np.random.uniform(0, 2)).clip(22, 28)


family_size = np.random.choice(range(1, 8), size=num_records, p=[0.4, 0.3, 0.15, 0.1, 0.03, 0.015, 0.005])
income_level = np.random.choice(["Low", "Middle", "High"], size=num_records, p=[0.4, 0.4, 0.2])
marital_status = np.random.choice(["Single", "Married", "Divorced"], size=num_records, p=[0.7, 0.25, 0.05])


major = np.random.choice(["STEM", "Business", "Arts", "Education", "Health Sciences"], size=num_records, p=[0.3, 0.2, 0.2, 0.15, 0.15])
study_hours = np.random.normal(loc=15, scale=5, size=num_records).clip(0, 40)
student_loan_amount = np.random.normal(loc=20000, scale=10000, size=num_records).clip(0, 50000)
campus_engagement = np.random.choice(["Low", "Medium", "High"], size=num_records, p=[0.4, 0.4, 0.2])
first_gen_student = np.random.choice([True, False], size=num_records, p=[0.3, 0.7])
enrollment_status = np.random.choice(["Full-Time", "Part-Time"], size=num_records, p=[0.8, 0.2])
distance_from_home = np.random.normal(loc=50, scale=30, size=num_records).clip(0, 200)
work_hours = np.random.normal(loc=10, scale=5, size=num_records).clip(0, 30)


institution_type = np.random.choice(["Public", "Private"], size=num_records, p=[0.7, 0.3])
support_utilization = ((4.0 - gpa) / 4.0 + np.random.normal(0, 0.2, num_records)).clip(0, 1)
life_events = np.random.choice(
    ["None", "Health Issues", "Family Issues", "Financial Problems"],
    size=num_records,
    p=[0.6, 0.2, 0.15, 0.05]
)


data = pd.DataFrame({
    "Student_ID": range(1, num_records + 1),
    "GPA": gpa.round(2),
    "SAT_Score": sat_scores.astype(int),
    "ACT_Score": act_scores,
    "Family_Size": family_size,
    "Income_Level": income_level,
    "Marital_Status": marital_status,
    "Support_Center_Utilization": support_utilization.round(2),
    "Retention_Rate": retention_rate.round(2),
    "Graduation_Rate": graduation_rate.round(2),
    "Life_Event": life_events,
    "Institution_Type": institution_type,
    "Graduation_Age": graduation_age.round(1),
    "Major": major,
    "Study_Hours_Per_Week": study_hours.round(1),
    "Student_Loan_Amount": student_loan_amount.round(2),
    "Campus_Engagement": campus_engagement,
    "First_Gen_Student": first_gen_student,
    "Enrollment_Status": enrollment_status,
    "Distance_From_Home": distance_from_home.round(1),
    "Work_Hours_Per_Week": work_hours.round(1)
})


data.to_csv("student_success_enhanced.csv", index=False)

print("Enhanced dataset saved to 'student_success_enhanced.csv'.")
