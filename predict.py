import joblib
import pandas as pd

model = joblib.load("model.pkl")

sample = {
    "Gender": [1],
    "Married": [1],
    "Education": [0],
    "Self_Employed": [0],
    "ApplicantIncome": [5000],
    "LoanAmount": [150],
    "Credit_History": [1],
    "Property_Area": [2]
}

df = pd.DataFrame(sample)
prediction = model.predict(df)[0]
print("✅ Loan Status:", "Approved" if prediction else "Rejected")
