# Disease Prediction Using Machine Learning

![GitHub stars](https://img.shields.io/github/stars/your-username/disease-prediction-ml?style=social)
![GitHub forks](https://img.shields.io/github/forks/your-username/disease-prediction-ml?style=social)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 **Overview**
This project aims to predict **42 different types of diseases** based on **132 medical parameters** using machine learning techniques. The model assists physicians in diagnosing diseases accurately and efficiently by analyzing patient symptoms and medical history.

The project includes:
- **Exploratory Data Analysis (EDA)** to understand the dataset.
- **Data Preprocessing** to handle missing values, encode categorical variables, and normalize features.
- **Model Training** using **Decision Tree, Random Forest, and SVM** classifiers.
- **Evaluation** of model performance using accuracy, precision, recall, and F1-score.
- **Prediction** on new patient data.

---

## 📂 **Project Structure**

```
disease-prediction-ml/

│

├── data/

│   ├── Training_data.csv       # Training dataset (4920 samples, 133 features)

│   └── Testing_data.csv        # Testing dataset (42 samples, 132 features)

│

├── notebooks/

│   └── disease_prediction.ipynb # Jupyter Notebook for EDA, preprocessing, training, and evaluation

│

├── models/

│   └── disease_prediction_model.pkl # Saved trained model (Decision Tree)

│

├── src/

│   ├── prediction.py            # Script for making predictions on new data

│   └── utils.py                 # Utility functions (e.g., preprocessing)

│

├── requirements.txt             # Python dependencies

├── README.md                    # Project documentation

└── .gitignore                   # Files to ignore in Git
```

---

## 🛠 **Installation**
### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook (for running the notebook)
- Required Python libraries (listed in `requirements.txt`)

### Steps
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/disease-prediction-ml.git
   cd disease-prediction-ml

2. **Install dependencies**:
   ```
   pip install -r requirements.txt
   ```

3. **Open the Jupyter Notebook**:

  ```
  jupyter notebook
  ```

Navigate to the notebooks/ folder and open disease_prediction.ipynb.

---------

## 🚀 Usage
1. Exploratory Data Analysis (EDA)

Open disease_prediction.ipynb and run the cells in Step 1 to explore the dataset.
Visualize feature distributions, correlations, and target variable distributions.
2. Data Preprocessing

Run the cells in Step 2 to:

Handle missing values.
Encode categorical variables.
Normalize numerical features.
Split the data into training and testing sets.

3. Model Training

Run the cells in Step 3 to train the following models:

Decision Tree
Random Forest
SVM (Support Vector Machine)

The best model is saved as disease_prediction_model.pkl in the models/ folder.
4. Model Evaluation

Run the cells in Step 4 to evaluate the models using:

Accuracy
Classification report (precision, recall, F1-score)
Confusion matrix

5. Making Predictions

Use the prediction.py script to predict diseases for new patient data:
```
python src/prediction.py
```
-Ensure the new data is in the same format as the training data (132 features, no target column).

-----------


## 📊 Dataset
Description

Source: Kaggle - Disease Prediction Dataset
Features: 132 medical parameters (symptoms, medical history, etc.)
Target: 42 disease types (e.g., Fungal infection, Allergy, Diabetes, etc.)
Training Data: 4920 samples
Testing Data: 42 samples
Features
The dataset includes symptoms and medical observations such as:

itching, skin_rash, nodal_skin_eruptions
fever, cough, headache
fatigue, weight_loss, anxiety
And 129 other parameters.
Target Variable

prognosis: The predicted disease (42 unique classes).

------------------

## 🎯 Models Used

| Model | Accuracy (Expected) | Description |
| --- | --- | --- |
| Decision Tree | ~98% | Simple, interpretable model for classification. |
| Random Forest | ~99% | Ensemble of decision trees for improved accuracy. |
| SVM (Support Vector Machine) | ~97% | Effective for high-dimensional data. |

-------------------

## 📈 Results
Example Output

Decision Tree Accuracy: 0.9876
Classification Report:
              precision    recall  f1-score   support
   Fungal infection       1.00      1.00      1.00        10
   Allergy                1.00      1.00      1.00        10
   ...

   # Confusion Matrix

    A confusion matrix is generated for each model to visualize performance.
----------------------

## 🤝 Contributing

Contributions are welcome! Here’s how you can contribute:

1. Fork the repository and create a new branch.
2. Make your changes (e.g., add new models, improve preprocessing, or fix bugs).
3. Submit a pull request with a clear description of your changes.

 
**Areas for Improvement**

- Experiment with deep learning models (e.g., Neural Networks).
- Optimize hyperparameters using GridSearchCV or RandomizedSearchCV.
- Deploy the model as a web app (e.g., using Flask or FastAPI).
 
----------------------

## 📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

 --------------------------

 ## 📧 Contact
 For questions or feedback, feel free to reach out:

- Email: meroo468saggaf@gmail.com
- GitHub: @MEROO1010
- LinkedIn: [(https://www.linkedin.com/in/marwah-alsaggaf-ba44011b2/)]


        
