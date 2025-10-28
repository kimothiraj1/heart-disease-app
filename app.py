import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os # Import os to check for file existence

# --- Load the Model ---
# Load the pre-trained pipeline (preprocessor + model)
model_filename = 'heart_disease_model.pkl'

if not os.path.exists(model_filename):
    st.error(f"Model file '{model_filename}' not found. "
             "Please run the `heart_disease_model_training.py` script first to generate the model file.")
    st.stop()
else:
    try:
        model = joblib.load(model_filename)
    except Exception as e:
        st.error(f"An error occurred while loading the model: {e}")
        st.stop()

# --- Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Predictor",
    page_icon="❤️",
    layout="wide"
)

# --- Application Title ---
st.title("❤️ Heart Disease Predictor")
st.markdown("This app uses a Logistic Regression model to predict the likelihood of a patient having heart disease based on their clinical data.")

# --- Define Input Mappings ---
# These are based on the original dataset and the preprocessing in the notebook
sex_map = {'Female': 0, 'Male': 1}
fbs_map = {'False (<= 120 mg/dl)': 0, 'True (> 120 mg/dl)': 1}
exang_map = {'No': 0, 'Yes': 1}

# These are the exact categories the OneHotEncoder was trained on
# From the notebook, we saw these are the most common values.
# The 'imputer' in the pipeline will handle any new/unseen values if they were
# to somehow appear, but the 'thal' categories are well-defined.
cp_categories = ['asymptomatic', 'atypical angina', 'non-anginal', 'typical angina']
restecg_categories = ['lv hypertrophy', 'normal', 'st-t abnormality']
slope_categories = ['downsloping', 'flat', 'upsloping']
thal_categories = ['fixed defect', 'normal', 'reversable defect']


# --- Sidebar for User Inputs ---
st.sidebar.header("Patient Information")

with st.sidebar:
    # --- Create columns for a cleaner layout ---
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=50)
        sex = st.selectbox("Sex", options=list(sex_map.keys()), index=1)
        cp = st.selectbox("Chest Pain Type (cp)", options=cp_categories, index=0)
        trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=80, max_value=220, value=120)
        chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
        fbs = st.selectbox("Fasting Blood Sugar (fbs)", options=list(fbs_map.keys()), index=0)

    with col2:
        restecg = st.selectbox("Resting ECG (restecg)", options=restecg_categories, index=1)
        thalch = st.number_input("Max Heart Rate (thalch)", min_value=60, max_value=220, value=150)
        exang = st.selectbox("Exercise Induced Angina (exang)", options=list(exang_map.keys()), index=0)
        oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
        slope = st.selectbox("Slope of ST Segment (slope)", options=slope_categories, index=1)
        # In the notebook, 'ca' was imputed with the median (0.0). We'll use 0-4 as a reasonable range.
        ca = st.number_input("Number of Major Vessels (ca)", min_value=0, max_value=4, value=0)
        thal = st.selectbox("Thallium Stress Test (thal)", options=thal_categories, index=1)

# --- Create DataFrame for Prediction ---
# This DataFrame must have the exact column names and order as the one used for training
# The pipeline will handle imputation for any 'None' or 'NaN' values if they are passed
# (e.g., if we didn't provide a value for trestbps), but our form ensures all values are provided.
input_data = {
    'age': [age],
    'sex': [sex_map[sex]],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs_map[fbs]],
    'restecg': [restecg],
    'thalch': [thalch],
    'exang': [exang_map[exang]],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [float(ca)], # Ensure 'ca' is float, as it was in training
    'thal': [thal]
}
input_df = pd.DataFrame(input_data)

# --- Main Page for Prediction and Results ---
st.subheader("Prediction")

# "Predict" button
if st.button("Predict Likelihood of Heart Disease", type="primary"):
    if 'model' in locals():
        # Make prediction
        try:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0]

            # Get the probability of the positive class (heart disease)
            prob_heart_disease = probability[1]

            # Display the result
            st.subheader("Result")
            if prediction == 1:
                st.error(f"**High Risk of Heart Disease**", icon="💔")
                st.markdown(f"The model predicts a **{prob_heart_disease*100:.2f}%** probability of heart disease.")
            else:
                st.success(f"**Low Risk of Heart Disease**", icon="❤️")
                st.markdown(f"The model predicts a **{prob_heart_disease*100:.2f}%** probability of heart disease.")

            # Show details
            with st.expander("Show Prediction Details"):
                st.write("Based on the input data, the model made the following prediction:")
                st.write(f"**Prediction (0 = No Disease, 1 = Disease):** `{prediction}`")
                st.write(f"**Probability (No Disease):** `{probability[0]*100:.2f}%`")
                st.write(f"**Probability (Heart Disease):** `{probability[1]*100:.2f}%`")

            with st.expander("Show Input Data"):
                st.dataframe(input_df.T.rename(columns={0: 'Input Value'}))

        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Model is not loaded. Cannot perform prediction.")

st.markdown("---")
st.markdown(
    "<small>Disclaimer: This app is for educational purposes only and is not a substitute "
    "for professional medical advice, diagnosis, or treatment. "
    "Always seek the advice of your physician or other qualified health provider "
    "with any questions you may have regarding a medical condition.</small>",
    unsafe_allow_html=True
)

