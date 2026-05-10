import streamlit as st
import joblib
import pandas as pd

# Load the model and label encoder
model = joblib.load('../models/random_forest.pkl')
le = joblib.load('../models/label_encoder.pkl')

# Title
st.title('Disease Prediction App')
st.write('Enter patient symptoms to predict the disease.')

# Load symptom list (from the dataset)
symptoms = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue', 'muscle_wasting',
    # Add all 132 symptoms here
]

# Input form
st.header('Patient Symptoms')
symptom_inputs = {}
for symptom in symptoms:
    symptom_inputs[symptom] = st.selectbox(symptom, options=[0, 1], format_func=lambda x: 'No' if x == 0 else 'Yes')

# Predict button
if st.button('Predict Disease'):
    # Convert inputs to DataFrame
    input_data = pd.DataFrame([symptom_inputs])

    # Predict
    prediction = model.predict(input_data)
    predicted_disease = le.inverse_transform(prediction)[0]

    # Display result
    st.success(f'Predicted Disease: **{predicted_disease}**')