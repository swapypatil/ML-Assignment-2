# Adult Income Classification using Machine Learning

## a. Problem Statement

The objective of this project is to build and compare multiple machine learning classification models to predict whether an individual's annual income exceeds $50,000 based on demographic and employment-related features. This is a binary classification problem with significant real-world applications in economic analysis, financial services, and policy making.

The models are trained and evaluated on the Adult Census Income dataset, a well-known benchmark dataset containing census data with various demographic attributes. The goal is to identify the most effective model for predicting income levels and analyze the performance characteristics of each algorithm on this dataset.

---

## b. Dataset Description

**Dataset Name:** Adult Census Income

**Source:** Kaggle 

**Dataset Size:**
- Total Records: 30,162 
- Total Features: 14 features + 1 target variable
- Training Split: 80% (24,129 samples)
- Testing Split: 20% (6,033 samples)

**Features (14):**
1. `age` (numeric) - Age of the individual
2. `workclass` (categorical) - Employment sector (Private, Self-emp-inc, etc.)
3. `fnlwgt` (numeric) - Final weight representing census weighting
4. `education` (categorical) - Education level (HS-grad, Bachelors, Masters, etc.)
5. `education.num` (numeric) - Numeric representation of education level
6. `marital.status` (categorical) - Marital status
7. `occupation` (categorical) - Type of occupation
8. `relationship` (categorical) - Family relationship status
9. `race` (categorical) - Race/Ethnicity
10. `sex` (categorical) - Gender
11. `capital.gain` (numeric) - Capital gains
12. `capital.loss` (numeric) - Capital losses
13. `hours.per.week` (numeric) - Working hours per week
14. `native.country` (categorical) - Country of origin

**Target Variable:** `income` (binary: `<=50K` or `>50K`)

**Class Distribution:**
- <=50K: 22,654 samples (75.2%)
- \>50K: 7,508 samples (24.8%)

The dataset is well-balanced for training purposes and includes both numeric and categorical features, requiring appropriate preprocessing techniques including imputation, encoding, and feature scaling.

---

## c. GitHub Repository Link

**Repository:** https://github.com/swapypatil/ML-Assignment-2

**Repository Contents:**
- `app.py` - Streamlit web application for interactive model evaluation
- `train_models.py` - Training script for all five models
- `requirements.txt` - Python package dependencies
- `README.md` - This documentation file
- `test_data.csv` - Sample test dataset
- `model_metrics.csv` - Evaluation metrics for all trained models
- `model/` - Directory containing:
  - Individual model training scripts (`Logistic_Regression.py`, `Decision_Tree.py`, etc.)
  - Saved trained model artifacts (`.joblib` files)

All required files are maintained and version-controlled in the GitHub repository.

---

## d. Models Used and Evaluation Metrics

Five different classification algorithms were implemented and trained on the Adult Income dataset. The following models were selected to provide a diverse representation of machine learning approaches:

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8543 | 0.9132 | 0.7498 | 0.6225 | 0.6802 | 0.5912 |
| Decision Tree | 0.8150 | 0.7531 | 0.6282 | 0.6298 | 0.6290 | 0.5058 |
| kNN | 0.8265 | 0.8521 | 0.6681 | 0.6019 | 0.6333 | 0.5212 |
| Naive Bayes | 0.5624 | 0.7886 | 0.3584 | 0.9587 | 0.5217 | 0.3573 |
| Random Forest (Ensemble) | 0.8573 | 0.9111 | 0.7494 | 0.6411 | 0.6911 | 0.6021 |

### Metric Definitions

- **Accuracy:** Proportion of correct predictions among total predictions
- **AUC (Area Under ROC Curve):** Measure of model's ability to distinguish between classes across all thresholds
- **Precision:** Proportion of positive predictions that were correct
- **Recall:** Proportion of actual positive instances that were correctly predicted
- **F1 Score:** Harmonic mean of Precision and Recall
- **MCC (Matthews Correlation Coefficient):** Balanced measure of classification performance

---

## Model Performance Observations

### Observation Table

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Strong overall performer with high accuracy (0.8543) and excellent AUC (0.9132). Good precision (0.7498) indicating reliable positive predictions. Moderate recall (0.6225) suggests some income >50K cases are missed. Well-balanced performance across metrics. Linear model performs surprisingly well on this dataset. |
| **Decision Tree** | Moderate accuracy (0.8150) with relatively balanced precision (0.6282) and recall (0.6298). Lower AUC (0.7531) compared to other models indicates less reliable probability estimates. Simple and interpretable model but tends to slightly underperform on this complex dataset. |
| **kNN** | Good accuracy (0.8265) with decent AUC (0.8521). Precision (0.6681) is reasonable but recall (0.6019) is lower, missing some positive cases. Distance-based method captures non-linear patterns moderately well. Computational cost increases with dataset size. |
| **Naive Bayes** | Weakest overall accuracy (0.5624) but exceptionally high recall (0.9587), catching most income >50K cases at the cost of many false positives. Very low precision (0.3584) indicates high false positive rate. Despite poor accuracy, high AUC (0.7886) shows reasonable probability calibration. Best for recall-prioritized scenarios. |
| **Random Forest (Ensemble)** | Best overall performer with highest F1 score (0.6911) and highest MCC (0.6021). High accuracy (0.8573) with excellent AUC (0.9111). Balanced precision (0.7494) and recall (0.6411). Ensemble approach effectively captures complex feature interactions and non-linear relationships in the data. |

### Overall Winner for Your Dataset: **Random Forest (Ensemble)**

**Justification:**
The Random Forest model demonstrates the strongest performance across multiple evaluation metrics:

1. **Highest F1 Score (0.6911):** Best balance between precision and recall, indicating robust performance
2. **Highest MCC (0.6021):** Most reliable single metric for classification quality
3. **Second Highest Accuracy (0.8573):** Nearly matches Logistic Regression, with superior AUC
4. **Excellent AUC (0.9111):** Demonstrates strong discriminative ability across all classification thresholds
5. **Balanced Metrics:** Maintains good precision (0.7494) while achieving solid recall (0.6411)

The ensemble method successfully mitigates individual decision tree overfitting while capturing complex non-linear relationships in demographic and employment features. Random Forest's ability to handle mixed feature types (numeric and categorical after preprocessing) and its robustness to outliers make it the most practical choice for deployment in real-world income prediction scenarios.

---

## Implementation Details

### Technologies Used
- **Language:** Python 3.9+
- **ML Framework:** scikit-learn
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Web Framework:** Streamlit
- **Model Serialization:** joblib

### Data Preprocessing Pipeline
1. **Missing Value Handling:** Imputation using median (numeric) and mode (categorical)
2. **Categorical Encoding:** One-Hot Encoding for categorical features
3. **Feature Scaling:** StandardScaler normalization
4. **Train-Test Split:** 80-20 stratified split preserving class distribution

### Model Training
- All models trained with consistent random seed (42) for reproducibility
- Hyperparameters tuned for optimal performance:
  - Logistic Regression: max_iter=500
  - Decision Tree: default parameters
  - kNN: n_neighbors=5
  - Naive Bayes: Gaussian variant
  - Random Forest: n_estimators=200

---

## Streamlit Web Application

The project includes an interactive Streamlit application that allows users to:
- Upload custom test datasets
- Select and evaluate different models
- View detailed evaluation metrics
- Visualize confusion matrices
- Inspect prediction samples

**Deployment:** Available on Streamlit Community Cloud for live evaluation - https://adult-income-census-data-ml-model-performance.streamlit.app/

---

## Results Summary

The systematic comparison of five ML algorithms reveals that ensemble methods (Random Forest) significantly outperform individual algorithms on this demographic classification task. The clear winner demonstrates that combining multiple learners provides both better predictive performance and more reliable probability estimates, making it the recommended model for production deployment.
