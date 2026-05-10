# Exploratory Data Analysis (EDA) Report

## Dataset Overview
- **Training Data Shape**: (4920, 133)  # Example: 4920 rows, 133 columns (132 features + 1 target)
- **Testing Data Shape**: (1230, 132)    # Example: 1230 rows, 132 features
- **Target Variable**: `prognosis` (42 unique diseases)

## Missing Values
- No missing values found in the training or testing datasets.

## Disease Distribution
- The dataset contains **42 unique diseases**.
- The most common disease is **[Disease Name]** with **[Count]** cases.
- The least common disease is **[Disease Name]** with **[Count]** cases.
- Visualization saved to `reports/visualizations/disease_distribution.png`.

## Feature Analysis
- All features are **binary (0 or 1)**, representing the presence or absence of symptoms.
- Example: `itching`, `skin_rash`, `nodal_skin_eruptions`, etc.