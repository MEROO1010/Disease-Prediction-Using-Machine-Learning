import joblib
import pandas as pd

def load_model_and_objects():
    """Load the trained model and preprocessing objects."""
    model = joblib.load('../models/random_forest.pkl')
    le = joblib.load('../models/label_encoder.pkl')
    return model, le

def predict_disease(input_data):
    """
    Predict disease for new input data.
    Args:
        input_data: DataFrame of new input data (features only).
    Returns:
        Predicted disease.
    """
    model, le = load_model_and_objects()
    prediction = model.predict(input_data)
    predicted_disease = le.inverse_transform(prediction)[0]
    return predicted_disease

# Example usage
if __name__ == "__main__":
    # Example input (replace with actual data)
    input_data = pd.DataFrame({
        'itching': [1],
        'skin_rash': [1],
        'nodal_skin_eruptions': [0],
        # Add all 132 features here (set to 0 or 1)
    })

    predicted_disease = predict_disease(input_data)
    print(f"Predicted Disease: {predicted_disease}")