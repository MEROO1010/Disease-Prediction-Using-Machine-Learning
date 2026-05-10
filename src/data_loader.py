import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_data(train_path, test_path):
    """Load training and testing datasets."""
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    return train_data, test_data

def preprocess_data(train_data, test_data=None, save_objects=True):
    """
    Preprocess the data:
    - Encode the target variable ('prognosis').
    - Split into training and validation sets.
    - Standardize features (optional for tree-based models).
    """
    # Encode the target variable
    le = LabelEncoder()
    train_data['prognosis_encoded'] = le.fit_transform(train_data['prognosis'])

    # Separate features and target
    X = train_data.drop(['prognosis', 'prognosis_encoded'], axis=1)
    y = train_data['prognosis_encoded']

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize features (optional)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Save preprocessing objects
    if save_objects:
        joblib.dump(le, '../models/label_encoder.pkl')
        joblib.dump(scaler, '../models/scaler.pkl')

    return X_train, X_val, y_train, y_val, le, scaler