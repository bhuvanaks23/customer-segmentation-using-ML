# customer-segmentation-using-ML
Customer Segmentation for Targeted Marketing Using Machine Learning

Overview

A multi-class classification project that predicts customer segments (A, B, C, D) for an automobile company expanding into new markets. The goal is to apply a successful segmentation-based marketing strategy from existing markets to new customer bases by classifying customers using their demographic and behavioral attributes.

Built as part of MSc in Information Systems — University College Dublin (2024–2025)

Dataset
Source: Kaggle — Customer Segmentation Dataset
Size: ~8,068 customer records, 11 features
Features: Gender, Age, Profession, Work Experience, Spending Score, Family Size, Marital Status, Graduation Status
Target Variable: Customer Segment (A / B / C / D)

Approach
1. Exploratory Data Analysis
Analyzed class distribution across the four segments
Identified outliers in Age, Work Experience, and Family Size using boxplots
Examined feature distributions for numerical and categorical variables separately

3. Data Preprocessing
Imputed missing values — median for numerical features, mode for categorical
Applied Label Encoding to convert categorical features to numerical
Standardized features using StandardScaler
Split data into training and validation sets

4. Model Training & Comparison

Trained and compared six classification models:

Model	Notes
Logistic Regression	Simple baseline; struggled with non-linear patterns
Decision Tree	Overfitted on training data
Random Forest	Good fit but showed signs of overfitting
SVM	Consistent train/validation performance
Naive Bayes	Underperformed due to simplifying assumptions
XGBoost	Best performer — selected as final model

4. Evaluation & Tuning
Primary Metric: Weighted F1-Score (chosen over accuracy due to multi-class setting)
Hyperparameter Tuning: GridSearchCV on XGBoost
Error Analysis: Reviewed misclassified samples — Segment B was consistently hardest to predict, likely due to overlapping feature patterns with other segments
Analyzed confusion matrices and classification reports per class

5. Key Findings
Segments C and D had the most distinguishable behavioral signatures
Segment B showed low feature separability across all models
Most influential features: Spending Score, Profession, Age
The available feature set limits how well segments can be distinguished — richer features (purchase history, location data) would likely improve performance


Tech Stack
Language: Python
Libraries: pandas, NumPy, scikit-learn, XGBoost, seaborn, matplotlib
Tools: Jupyter Notebook, GridSearchCV
Repository Structure
├── Customer_Segmentation.ipynb   # Full analysis notebook
├── Customer_Data.csv             # Dataset
└── README.md


Future Improvements
Apply SHAP for model interpretability and stakeholder communication
Engineer additional features or source richer customer data
Explore ensemble stacking or neural network approaches
Conduct fairness audits on demographic features before deployment


References
Buya et al. (2020) — Multinomial Logistic Regression for land-use modelling
Wang (2024) — Customer Segmentation Based on Machine Learning Methods
Shrestha & Shakya (2022) — Customer Churn Prediction using XGBoost
