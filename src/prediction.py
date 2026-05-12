import joblib
import pandas as pd
from utils import preprocess_data

# Load the trained model
model = joblib.load("../models/disease_prediction_model.pkl")

# Load new data for prediction (replace with your new data file)
new_data = pd.read_csv("../data/new_patient_data.csv")  # Ensure this file exists

# Preprocess the new data
X_new, _ = preprocess_data(new_data, target_column="target_column")  # Replace "target_column" with the actual column name

# Make predictions
predictions = model.predict(X_new)

# Display predictions
print("Predictions for new data:")
for i, pred in enumerate(predictions):
    print(f"Patient {i + 1}: Disease Type {pred}")