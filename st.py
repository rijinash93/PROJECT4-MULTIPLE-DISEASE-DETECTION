import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Multiple Disease Prediction", layout="wide")
st.title("🩺 Multiple Disease Prediction System")

# ====================== Load Models ======================
try:
    kidney_model = joblib.load("kidney_rf_model.pkl")
except:
    kidney_model = None
try:
    liver_model = joblib.load("liver_model.pkl")
except:
    liver_model = None
try:
    parkinson_model = joblib.load("parkinsons_model.pkl")
except:
    parkinson_model = None


# ====================== Helper Function ======================
def fill_missing_features(input_df, expected_cols):
    """Ensure all expected columns are present, add missing with 0."""
    for col in expected_cols:
        if col not in input_df.columns:
            input_df[col] = 0
    return input_df[expected_cols]


# ============================================================
# 1️⃣ Kidney Disease Prediction
# ============================================================
st.header("Kidney Disease Prediction")

age = st.number_input("Age", 0, 120, 45)
bp = st.number_input("Blood Pressure (bp)", 0, 200, 80)
sg = st.number_input("Specific Gravity (sg)", 1.0, 1.03, 1.02)
al = st.number_input("Albumin (al)", 0, 5, 1)
su = st.number_input("Sugar (su)", 0, 5, 0)

kidney_input = pd.DataFrame([[age, bp, sg, al, su]], columns=['age', 'bp', 'sg', 'al', 'su'])

expected_kidney_cols = [
    'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr',
    'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
    'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
]
kidney_input = fill_missing_features(kidney_input, expected_kidney_cols)

if st.button("🔍 Predict Kidney Disease"):
    if kidney_model:
        pred = kidney_model.predict(kidney_input)[0]
        prob = kidney_model.predict_proba(kidney_input)[0][1]
        st.success(f"Prediction: {'CKD' if pred == 1 else 'Not CKD'}")
        st.info(f"Probability of CKD: {prob*100:.2f}%")
    else:
        st.error("Kidney model not found!")


# ============================================================
# 2️⃣ Liver Disease Prediction
# ============================================================
st.header("Liver Disease Prediction")

age = st.number_input("Age (Years)", 0, 120, 50)
gender = st.selectbox("Gender", ["Male", "Female"])
tb = st.number_input("Total Bilirubin", 0.0, 75.0, 0.7)
db = st.number_input("Direct Bilirubin", 0.0, 19.0, 0.1)
alp = st.number_input("Alkaline Phosphotase", 0, 2000, 200)
alt = st.number_input("Alamine Aminotransferase", 0, 2000, 30)
ast = st.number_input("Aspartate Aminotransferase", 0, 2000, 35)
tp = st.number_input("Total Proteins", 0.0, 10.0, 6.8)
alb = st.number_input("Albumin", 0.0, 6.0, 3.3)
agr = st.number_input("Albumin and Globulin Ratio", 0.0, 3.0, 1.0)

liver_input = pd.DataFrame([[age, gender, tb, db, alp, alt, ast, tp, alb, agr]],
    columns=['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
             'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
             'Aspartate_Aminotransferase', 'Total_Protiens',
             'Albumin', 'Albumin_and_Globulin_Ratio']
)

# Convert categorical
liver_input['Gender'] = 1 if liver_input['Gender'][0] == 'Male' else 0

expected_liver_cols = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
    'Alamine_Aminotransferase', 'Aspartate_Aminotransferase',
    'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'
]
liver_input = fill_missing_features(liver_input, expected_liver_cols)

if st.button("🔍 Predict Liver Disease"):
    if liver_model:
        pred = liver_model.predict(liver_input)[0]
        prob = liver_model.predict_proba(liver_input)[0][1]
        st.success(f"Prediction: {'Liver Disease' if pred == 1 else 'Healthy Liver'}")
        st.info(f"Probability: {prob*100:.2f}%")
    else:
        st.error("Liver model not found!")


# ============================================================
# 3️⃣ Parkinson’s Disease Prediction
# ============================================================
# ============================================================
# 3️⃣ Parkinson’s Disease Prediction (Full Feature Set)
# ============================================================
st.header("Parkinson’s Disease Prediction")

# Collect all 22 important features (based on common Parkinson dataset)
fo = st.number_input("MDVP:Fo(Hz)", 50.0, 300.0, 120.0)
fhi = st.number_input("MDVP:Fhi(Hz)", 60.0, 400.0, 150.0)
flo = st.number_input("MDVP:Flo(Hz)", 50.0, 300.0, 80.0)
jitter_percent = st.number_input("MDVP:Jitter(%)", 0.0, 1.0, 0.005)
jitter_abs = st.number_input("MDVP:Jitter(Abs)", 0.0, 0.01, 0.00007)
rap = st.number_input("MDVP:RAP", 0.0, 1.0, 0.0037)
ppq = st.number_input("MDVP:PPQ", 0.0, 1.0, 0.0055)
ddp = st.number_input("Jitter:DDP", 0.0, 1.0, 0.011)
shimmer = st.number_input("MDVP:Shimmer", 0.0, 1.0, 0.043)
shimmer_db = st.number_input("MDVP:Shimmer(dB)", 0.0, 1.0, 0.43)
apq3 = st.number_input("Shimmer:APQ3", 0.0, 1.0, 0.021)
apq5 = st.number_input("Shimmer:APQ5", 0.0, 1.0, 0.031)
apq = st.number_input("MDVP:APQ", 0.0, 1.0, 0.03)
dda = st.number_input("Shimmer:DDA", 0.0, 1.0, 0.065)
nhr = st.number_input("NHR", 0.0, 1.0, 0.022)
hnr = st.number_input("HNR", 0.0, 50.0, 21.0)
rpde = st.number_input("RPDE", 0.0, 1.0, 0.45)
dfa = st.number_input("DFA", 0.0, 1.0, 0.8)
spread1 = st.number_input("spread1", -10.0, 0.0, -4.5)
spread2 = st.number_input("spread2", 0.0, 1.0, 0.3)
d2 = st.number_input("D2", 0.0, 5.0, 2.3)
ppe = st.number_input("PPE", 0.0, 1.0, 0.28)

# Create input DataFrame
parkinson_input = pd.DataFrame([[
    fo, fhi, flo, jitter_percent, jitter_abs, rap, ppq, ddp,
    shimmer, shimmer_db, apq3, apq5, apq, dda, nhr, hnr,
    rpde, dfa, spread1, spread2, d2, ppe
]], columns=[
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
    'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
    'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
    'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
])

if st.button("🔍 Predict Parkinson’s Disease"):
    if parkinson_model:
        try:
            pred = parkinson_model.predict(parkinson_input)[0]
            prob = parkinson_model.predict_proba(parkinson_input)[0][1]
            st.success(f"Prediction: {'Parkinson’s Disease' if pred == 1 else 'Healthy'}")
            st.info(f"Probability: {prob*100:.2f}%")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.error("Parkinson model not found!")
