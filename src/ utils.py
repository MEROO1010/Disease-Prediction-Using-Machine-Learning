import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(data, target_column):
    """
    Preprocess the data:
    - Handle missing values.
    - Encode categorical variables.
    - Normalize numerical features.
    """
    # Handle missing values for numerical columns
    numerical_cols = data.select_dtypes(include=["int64", "float64"]).columns
    data[numerical_cols] = data[numerical_cols].fillna(data[numerical_cols].mean())

    # Handle missing values for categorical columns
    categorical_cols = data.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for col in categorical_cols:
        data[col] = label_encoder.fit_transform(data[col])

    # Normalize numerical features
    scaler = StandardScaler()
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

    # Split into features (X) and target (y)
    X = data.drop(target_column, axis=1)
    y = data[target_column]

    return X, yS