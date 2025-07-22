import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Load the trained model
# This model expects 'fnlwgt' as an input feature.
model = joblib.load("best_model.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary Classification App")
st.markdown("Predict whether an employee earns >50K or ≤50K based on input features.")

st.sidebar.header("Input Employee Details")

# Numerical Inputs (fnlwgt is NOT created as a direct sidebar input here)
age = st.sidebar.slider("Age", 17, 75, 30)
educational_num = st.sidebar.slider("Educational Number (Years of Education)", 5, 16, 9)
capital_gain = st.sidebar.number_input("Capital Gain", 0, 100000, 0)
capital_loss = st.sidebar.number_input("Capital Loss", 0, 4000, 0)
hours_per_week = st.sidebar.slider("Hours per week", 1, 99, 40)

# Define a default value for fnlwgt.
# You can replace this with a more representative value from your dataset
# if you know the mean or median fnlwgt.
# For example, if the median fnlwgt from your data is 178356.
default_fnlwgt = 178356 # A common central value, obtained from analyzing the dataset.

# Categorical Inputs and their options
workclass_options = ['Private', 'Self-emp-not-inc', 'Local-gov', 'Others', 'State-gov', 'Self-emp-inc', 'Federal-gov']
workclass = st.sidebar.selectbox("Workclass", workclass_options)

marital_status_options = ['Never-married', 'Married-civ-spouse', 'Divorced', 'Widowed', 'Separated', 'Married-spouse-absent', 'Married-AF-spouse']
marital_status = st.sidebar.selectbox("Marital Status", marital_status_options)

occupation_options = ['Prof-specialty', 'Craft-repair', 'Exec-managerial', 'Adm-clerical', 'Sales', 'Other-service', 'Machine-op-inspct', 'Others', 'Transport-moving', 'Handlers-cleaners', 'Farming-fishing', 'Tech-support', 'Protective-serv', 'Priv-house-serv', 'Armed-Forces']
occupation = st.sidebar.selectbox("Occupation", occupation_options)

relationship_options = ['Husband', 'Not-in-family', 'Own-child', 'Unmarried', 'Wife', 'Other-relative']
relationship = st.sidebar.selectbox("Relationship", relationship_options)

race_options = ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']
race = st.sidebar.selectbox("Race", race_options)

gender_options = ['Male', 'Female']
gender = st.sidebar.selectbox("Gender", gender_options)

native_country_options = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Puerto-Rico', 'Canada', 'El-Salvador', 'India', 'Cuba', 'England', 'Jamaica', 'South', 'China', 'Italy', 'Dominican-Republic', 'Vietnam', 'Guatemala', 'Columbia', 'Poland', 'Japan', 'Portugal', 'Taiwan', 'Haiti', 'Iran', 'Ecuador', 'France', 'Nicaragua', 'Peru', 'Greece', 'Ireland', 'Hong', 'Thailand', 'Cambodia', 'Trinadad&Tobago', 'Laos', 'Yugoslavia', 'Outlying-US(Guam-USVI-etc)', 'Hungary', 'Honduras', 'Scotland', 'Holand-Netherlands']
native_country = st.sidebar.selectbox("Native Country", native_country_options)

# --- Data for Display (human-readable, without fnlwgt) ---
# fnlwgt is intentionally omitted here for display purposes
input_display_data = {
    'age': age,
    'workclass': workclass,
    'educational-num': educational_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}

st.write("### 🔎 Input Data (for your review)")
st.write(pd.DataFrame([input_display_data]))

# --- Data for Model Input (encoded, with fnlwgt) ---
# This dictionary MUST include fnlwgt as the model expects it,
# even if it's not a direct user input.
input_model_data = {
    'age': age,
    'workclass': workclass,
    'fnlwgt': default_fnlwgt, # fnlwgt is passed from our default_fnlwgt variable
    'educational-num': educational_num,
    'marital-status': marital_status,
    'occupation': occupation,
    'relationship': relationship,
    'race': race,
    'gender': gender,
    'capital-gain': capital_gain,
    'capital-loss': capital_loss,
    'hours-per-week': hours_per_week,
    'native-country': native_country
}

# Apply Label Encoding to categorical features for the model input
# This ensures consistency with how the model was trained.
encoders = {
    'workclass': (LabelEncoder(), workclass_options),
    'marital-status': (LabelEncoder(), marital_status_options),
    'occupation': (LabelEncoder(), occupation_options),
    'relationship': (LabelEncoder(), relationship_options),
    'race': (LabelEncoder(), race_options),
    'gender': (LabelEncoder(), gender_options),
    'native-country': (LabelEncoder(), native_country_options)
}

for col, (encoder, options) in encoders.items():
    encoder.fit(options) # Fit on all possible options
    # Transform the single value from the sidebar input
    input_model_data[col] = encoder.transform([input_model_data[col]])[0]

# Convert to DataFrame for the model
input_df_for_model = pd.DataFrame([input_model_data])

# Ensure columns are in the exact order the model expects
# This list MUST match the columns in your 'x' DataFrame that was used for training
expected_model_columns = [
    'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
    'occupation', 'relationship', 'race', 'gender', 'capital-gain',
    'capital-loss', 'hours-per-week', 'native-country'
]
input_df_for_model = input_df_for_model[expected_model_columns]

# The processed data for the model is now hidden by default
# st.write("### ⚙️ Processed Data (for Model Input - Hidden from view)")
# st.write(input_df_for_model) # Uncomment this line for debugging if needed

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df_for_model)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())

    # --- Apply Label Encoding to batch data ---
    for col, (encoder, options) in encoders.items():
        encoder = LabelEncoder()
        encoder.fit(options)
        if col in batch_data.columns:
            batch_data[col] = encoder.transform(batch_data[col])
        else:
            st.warning(f"Column '{col}' not found in uploaded batch data. Filling with default encoded value 0.")
            batch_data[col] = 0

    # --- Ensure fnlwgt is handled for batch data ---
    if 'fnlwgt' not in batch_data.columns:
        st.warning(f"Column 'fnlwgt' not found in uploaded batch data. Filling with default value {default_fnlwgt}.")
        batch_data['fnlwgt'] = default_fnlwgt
    
    # --- Ensure all required columns are present and in the correct order for the model ---
    # This list MUST match the columns in your 'x' DataFrame that was used for training
    expected_model_columns_batch = [
        'age', 'workclass', 'fnlwgt', 'educational-num', 'marital-status',
        'occupation', 'relationship', 'race', 'gender', 'capital-gain',
        'capital-loss', 'hours-per-week', 'native-country'
    ]

    # Check if all expected columns exist in batch_data, if not, add them with default values
    for col in expected_model_columns_batch:
        if col not in batch_data.columns:
            st.warning(f"Column '{col}' not found in uploaded batch data. Filling with 0.")
            batch_data[col] = 0
    
    # Reorder columns to match the training data
    batch_data = batch_data[expected_model_columns_batch]

    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head()) # This will show the encoded values in the batch prediction output
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
