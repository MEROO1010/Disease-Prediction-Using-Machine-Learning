import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_model(X_train, y_train, model_type='Random Forest', **kwargs):
    """
    Train a model based on the specified type.
    Args:
        X_train: Training features.
        y_train: Training labels.
        model_type: Type of model to train ('Decision Tree', 'Random Forest', 'Naive Bayes').
        **kwargs: Additional arguments for the model.
    Returns:
        Trained model.
    """
    models = {
        'Decision Tree': DecisionTreeClassifier(random_state=42, **kwargs),
        'Random Forest': RandomForestClassifier(random_state=42, **kwargs),
        'Naive Bayes': GaussianNB(**kwargs)
    }

    model = models.get(model_type)
    if model is None:
        raise ValueError(f"Unsupported model type: {model_type}")

    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_val, y_val):
    """
    Evaluate a model on validation data.
    Args:
        model: Trained model.
        X_val: Validation features.
        y_val: Validation labels.
    Returns:
        Dictionary of evaluation metrics.
    """
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, output_dict=True)
    cm = confusion_matrix(y_val, y_pred)

    return {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm
    }