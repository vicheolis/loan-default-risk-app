"""
Streamlit demo app for the Risk-Based Loan Approval Framework.
Loads the trained sklearn/imblearn pipeline (selected_loan_default_model.joblib)
and lets a user enter applicant details to get a default-probability score
plus a decision under three lending strategies.
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Loan Default Risk Predictor", page_icon="💳", layout="centered")

MODEL_PATH = "selected_loan_default_model.joblib"

STRATEGIES = {
    "Conservative": {"approve_threshold": 0.20, "reject_threshold": 0.50},
    "Balanced": {"approve_threshold": 0.30, "reject_threshold": 0.70},
    "Growth-Oriented": {"approve_threshold": 0.40, "reject_threshold": 0.80},
}


def decision(p, a, r):
    if p < a:
        return "Approve"
    if p <= r:
        return "Manual Review"
    return "Reject"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def badge(text):
    color = {"Approve": "green", "Manual Review": "orange", "Reject": "red"}.get(text, "gray")
    st.markdown(
        f"<span style='background-color:{color};color:white;padding:4px 10px;"
        f"border-radius:6px;font-weight:600'>{text}</span>",
        unsafe_allow_html=True,
    )


st.title("💳 Loan Default Risk Predictor")
st.caption(
    "Demo of a risk-based loan approval framework for digital banks, built with "
    "Logistic Regression + SMOTE on the Kaggle Loan Default Prediction Dataset."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error(
        f"Model file '{MODEL_PATH}' not found. Make sure it's in the same folder as app.py."
    )
    st.stop()

st.subheader("Applicant details")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    income = st.number_input("Annual income (RM)", min_value=0, value=60000, step=1000)
    loan_amount = st.number_input("Loan amount (RM)", min_value=0, value=15000, step=500)
    credit_score = st.number_input("Credit score", min_value=300, max_value=900, value=650)
    months_employed = st.number_input("Months employed", min_value=0, value=36)
with col2:
    num_credit_lines = st.number_input("Number of credit lines", min_value=0, value=3)
    interest_rate = st.number_input("Interest rate (%)", min_value=0.0, value=12.5, step=0.1)
    loan_term = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60], index=2)
    dti_ratio = st.slider("Debt-to-income ratio", 0.0, 1.0, 0.35, 0.01)

st.subheader("Applicant profile")
col3, col4 = st.columns(2)
with col3:
    education = st.selectbox("Education", ["High School", "Bachelor's", "Master's", "PhD"])
    employment_type = st.selectbox("Employment type", ["Full-time", "Part-time", "Self-employed", "Unemployed"])
    marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
    loan_purpose = st.selectbox("Loan purpose", ["Home", "Auto", "Education", "Business", "Other"])
with col4:
    has_mortgage = st.selectbox("Has mortgage?", ["Yes", "No"])
    has_dependents = st.selectbox("Has dependents?", ["Yes", "No"])
    has_cosigner = st.selectbox("Has co-signer?", ["Yes", "No"])

if st.button("Predict default risk", type="primary"):
    applicant = pd.DataFrame([{
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "LoanTerm": loan_term,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner,
    }])

    try:
        prob = model.predict_proba(applicant)[:, 1][0]
    except Exception as e:
        st.error(
            "Prediction failed. This usually means the input columns don't match what "
            f"the model was trained on. Details: {e}"
        )
        st.stop()

    st.subheader("Result")
    st.metric("Predicted probability of default", f"{prob:.1%}")

    st.write("Decision under each lending strategy:")
    for name, s in STRATEGIES.items():
        d = decision(prob, s["approve_threshold"], s["reject_threshold"])
        c1, c2 = st.columns([1, 1])
        with c1:
            st.write(f"**{name}**")
        with c2:
            badge(d)

st.divider()
st.caption(
    "Built for academic demonstration purposes (Master's coursework project). "
    "Not intended for real lending decisions."
)
