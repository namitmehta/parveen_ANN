from pathlib import Path
import pickle
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Churn Prediction (ANN)", layout="centered")
st.title("Bank Customer Churn Prediction (ANN)")

BASE_DIR = Path(__file__).resolve().parent

# ---------- OPTIONAL DEBUG (keep or remove) ----------
with st.expander("🔍 Debug: files in app folder"):
    st.write("BASE_DIR:", str(BASE_DIR))
    st.write("Files:", sorted([p.name for p in BASE_DIR.iterdir()]))

required = [
    "churn_model.h5",
    "scaler.pkl",
    "gender_encoder.pkl",
    "geo_encoder.pkl",
    "feature_columns.pkl",
]
missing = [f for f in required if not (BASE_DIR / f).exists()]
if missing:
    st.error(f"❌ Missing files in GitHub repo: {missing}")
    st.stop()
else:
    st.success("✅ All required files are present!")

# ---------- LOAD ARTIFACTS ----------
@st.cache_resource
def load_artifacts():
    model = load_model(BASE_DIR / "churn_model.h5")
    with open(BASE_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open(BASE_DIR / "gender_encoder.pkl", "rb") as f:
        gender_encoder = pickle.load(f)
    with open(BASE_DIR / "geo_encoder.pkl", "rb") as f:
        geo_encoder = pickle.load(f)
    with open(BASE_DIR / "feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, scaler, gender_encoder, geo_encoder, feature_columns

model, scaler, gender_encoder, geo_encoder, feature_columns = load_artifacts()

st.divider()

# ---------- UI INPUTS ----------
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=42)
tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)
balance = st.number_input("Balance", min_value=0.0, value=60000.0, step=1000.0)
num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_cr_card = st.selectbox("Has Credit Card", [0, 1], index=1)
is_active = st.selectbox("Is Active Member", [0, 1], index=1)
salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, step=1000.0)

# ---------- PREDICT ----------
if st.button("Predict Churn"):
    gender_val = gender_encoder.transform([gender])[0]

    geo_ohe = geo_encoder.transform(pd.DataFrame({"Geography": [geography]}))
    geo_cols = geo_encoder.get_feature_names_out(["Geography"])
    geo_df = pd.DataFrame(geo_ohe, columns=geo_cols)

    base_df = pd.DataFrame([{
        "CreditScore": credit_score,
        "Gender": gender_val,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary,
    }])

    final_df = pd.concat([base_df, geo_df], axis=1)
    final_df = final_df.reindex(columns=feature_columns, fill_value=0)

    x_scaled = scaler.transform(final_df)
    proba = float(model.predict(x_scaled, verbose=0)[0][0])

    st.subheader("Result")
    st.write(f"Churn Probability: **{proba:.4f}**")

    if proba >= 0.5:
        st.error("Customer is likely to churn ❌")
    else:
        st.success("Customer is not likely to churn ✅")
