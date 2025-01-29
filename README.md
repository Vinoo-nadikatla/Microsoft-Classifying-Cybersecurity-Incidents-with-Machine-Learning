# Microsoft-Classifying-Cybersecurity-Incidents-with-Machine-Learning
A machine learning pipeline classifies cybersecurity incidents as TP, BP, or FP using the Microsoft GUIDE dataset. It includes advanced preprocessing, data splitting, pipelines, a baseline model, advanced models, XGBoost optimization, SMOTE, and deployment-ready outputs with Python, scikit-learn, and imbalanced-learn.
Microsoft: Classifying Cybersecurity Incidents with Machine Learning
Overview
This repository contains the implementation of a machine learning pipeline designed to classify cybersecurity incidents into three categories: True Positive (TP), Benign Positive (BP), and False Positive (FP). Using the Microsoft GUIDE dataset, the project leverages advanced data preprocessing, feature engineering, and classification techniques to optimize model performance and support Security Operation Centers (SOCs) in automating incident triage.

Key Features
Extensive Data Preprocessing and Feature Engineering:

Null value handling and removal of irrelevant features.
Time-based feature extraction (day, hour, etc.) from timestamps.
Label encoding for categorical variables.
Feature correlation analysis to drop highly correlated features.
Machine Learning Model Training and Optimization:

Baseline models: Logistic Regression and Decision Trees.
Advanced models: Random Forest, Gradient Boosting, XGBoost, and LightGBM.
Techniques to handle class imbalance: SMOTE and class-weight adjustments.
Hyperparameter tuning using RandomizedSearchCV.
Model Evaluation:

Metrics: Macro-F1 score, precision, recall.
Comparison of models to select the best performer.
Deployment-Ready Solution:

Final model saved using joblib for easy deployment.

Business Use Cases
1. Security Operation Centers (SOCs)
Automate the triage process to prioritize critical threats efficiently.

2. Incident Response Automation
Enable systems to suggest appropriate actions for incident mitigation.

3. Threat Intelligence
Enhance detection capabilities using historical evidence and customer responses.

4. Enterprise Security Management
Reduce false positives and ensure timely addressing of true threats.

Dataset
The Microsoft GUIDE dataset provides comprehensive telemetry data across three hierarchies: evidence, alerts, and incidents. Key highlights include:

GUIDE_train.csv (2.43 GB) GUIDE_test.csv (1.09 GB) Kaggle Link to Dataset


Project Workflow
1. Data Preprocessing
Removed and handled missing and duplicate values, ensuring clean data.
Engineered features like Hour, Day, and Time from timestamps.
Encoded categorical features using LabelEncoder.
2. Exploratory Data Analysis (EDA)
Visualized incident distributions across Hour, Day,month and Category. EDA Visualizations EDA Visualizations EDA Visualizations EDA Visualizations

Identified significant class imbalance in target labels (TP, BP, FP). EDA Visualizations

Co-relation heatmap to understand co-linearity among the features EDA Visualizations

3. Model Training and Evaluation
Baseline Models: Logistic Regression and Decision Tree for initial benchmarks.

Advanced Models: Random Forest, Gradient Boosting, XGBoost, and LightGBM. Model Performance

Addressed class imbalance with SMOTE, improving F1-scores.

Selected XGBoost with the top 11 features for final evaluation.

Model Performance

4. Hyperparameter Tuning
Optimized XGBoost hyperparameters using RandomizedSearchCV.

Tuned parameters included max_depth, learning_rate, and n_estimators.

Hyperparameter Tuning

5. Feature Importance
Identified top features with SHAP, including OrgId, IncidentId, DetectorId, and more.

These features are used to improve computational efficiency and model accuracy.


6. Final Evaluation
Tested the final model on unseen data, achieving a high Macro-F1 Score.

Delivered a balanced and generalizable model for real-world applications.

Final Evaluation


cd <repository_directory>
Acknowledgments
Microsoft for providing the GUIDE dataset.
Open-source contributors of libraries and tools used in this project.
The data science and cybersecurity communities for inspiration and knowledge.
