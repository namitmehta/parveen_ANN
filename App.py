import pickle
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(page_title="Churn Prediction (ANN)", layout="centered")
st.title("Bank Customer Churn Prediction (ANN)")

@st.cache_resource
def load_artifacts():
    model = load_model("churn_model.h5")
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("gender_encoder.pkl", "rb") as f:
        gender_encoder = pickle.load(f)
    with open("geo_encoder.pkl", "rb") as f:
        geo_encoder = pickle.load(f)
    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)
    return model, scaler, gender_encoder, geo_encoder, feature_columns

model, scaler, gender_encoder, geo_encoder, feature_columns = load_artifacts()

# ---- UI Inputs ----
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

if st.button("Predict Churn"):
    # Build input row (raw)
    input_row = {
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary
    }

    # Gender encode (correct shape)
    gender_val = gender_encoder.transform([input_row["Gender"]])[0]

    # Geography one-hot (safe for new sklearn)
    geo_ohe = geo_encoder.transform(pd.DataFrame({"Geography": [input_row["Geography"]]}))
    geo_cols = geo_encoder.get_feature_names_out(["Geography"])
    geo_df = pd.DataFrame(geo_ohe, columns=geo_cols)

    # Combine into final dataframe
    base_df = pd.DataFrame([{
        "CreditScore": input_row["CreditScore"],
        "Gender": gender_val,
        "Age": input_row["Age"],
        "Tenure": input_row["Tenure"],
        "Balance": input_row["Balance"],
        "NumOfProducts": input_row["NumOfProducts"],
        "HasCrCard": input_row["HasCrCard"],
        "IsActiveMember": input_row["IsActiveMember"],
        "EstimatedSalary": input_row["EstimatedSalary"],
    }])

    final_df = pd.concat([base_df, geo_df], axis=1)

    # Ensure same feature order as training
    final_df = final_df.reindex(columns=feature_columns, fill_value=0)

    # Scale + predict
    x_scaled = scaler.transform(final_df)
    proba = float(model.predict(x_scaled, verbose=0)[0][0])

    st.subheader("Result")
    st.write(f"Churn Probability: **{proba:.4f}**")

    if proba >= 0.5:
        st.error("Customer is likely to churn ❌")
    else:
        st.success("Customer is not likely to churn ✅")
