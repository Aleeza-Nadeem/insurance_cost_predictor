import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Locate current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Medical Insurance Cost Predictor", page_icon="🏥"
)


@st.cache_resource
def load_artifacts():
  svr = joblib.load(os.path.join(BASE_DIR, 'svr_model.pkl'))
  scaler_X = joblib.load(os.path.join(BASE_DIR, 'scaler_X.pkl'))
  scaler_y = joblib.load(os.path.join(BASE_DIR, 'scaler_y.pkl'))
  columns = joblib.load(os.path.join(BASE_DIR, 'model_columns.pkl'))
  return svr, scaler_X, scaler_y, columns


svr, scaler_X, scaler_y, model_columns = load_artifacts()

st.title("🏥 Medical Insurance Cost Predictor")

col1, col2 = st.columns(2)
with col1:
  age = st.slider("Age", 18, 100, 30)
  bmi = st.number_input("BMI", 10.0, 50.0, 25.0, step=0.1)
  children = st.slider("Children", 0, 5, 0)

with col2:
  sex = st.selectbox("Sex", ["male", "female"])
  smoker = st.selectbox("Smoker", ["yes", "no"])
  region = st.selectbox(
      "Region", ["northeast", "northwest", "southeast", "southwest"]
  )

if st.button("Predict Insurance Cost", type="primary"):
  # Create input dictionary initialized with floats
  input_data = {col: 0.0 for col in model_columns}

  # Assign user values safely
  if 'age' in input_data:
    input_data['age'] = float(age)
  if 'bmi' in input_data:
    input_data['bmi'] = float(bmi)
  if 'children' in input_data:
    input_data['children'] = float(children)

  if 'sex_male' in input_data and sex == 'male':
    input_data['sex_male'] = 1.0
  if 'smoker_yes' in input_data and smoker == 'yes':
    input_data['smoker_yes'] = 1.0
  if f'region_{region}' in input_data:
    input_data[f'region_{region}'] = 1.0

  # Convert dictionary to DataFrame with exact column ordering
  input_df = pd.DataFrame([input_data])[model_columns]

  # Scale inputs and predict
  input_scaled = scaler_X.transform(input_df)
  pred_scaled = svr.predict(input_scaled)
  pred_log = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

  st.success(f"### Predicted Annual Charges: ${np.expm1(pred_log)[0]:,.2f}")
