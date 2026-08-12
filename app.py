import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Medical Insurance Cost Predictor", page_icon="🏥"
)


@st.cache_resource
def load_artifacts():
  svr = joblib.load('svr_model.pkl')
  scaler_X = joblib.load('scaler_X.pkl')
  scaler_y = joblib.load('scaler_y.pkl')
  columns = joblib.load('model_columns.pkl')
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
  input_df = pd.DataFrame(0, index=[0], columns=model_columns)
  input_df.loc[0, 'age'] = age
  input_df.loc[0, 'bmi'] = bmi
  input_df.loc[0, 'children'] = children

  if 'sex_male' in model_columns and sex == 'male':
    input_df.loc[0, 'sex_male'] = 1
  if 'smoker_yes' in model_columns and smoker == 'yes':
    input_df.loc[0, 'smoker_yes'] = 1
  if f'region_{region}' in model_columns:
    input_df.loc[0, f'region_{region}'] = 1

  input_scaled = scaler_X.transform(input_df)
  pred_scaled = svr.predict(input_scaled)
  pred_log = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()

  st.success(f"### Predicted Annual Charges: ${np.expm1(pred_log)[0]:,.2f}")
