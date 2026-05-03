# CDC Diabetes Health Indicators Analysis

## Project Overview
This project leverages the **CDC Behavioral Risk Factor Surveillance System (BRFSS)** dataset to predict diabetes status using 21 self-reported health indicators.

With diabetes affecting millions and incurring significant healthcare costs, this study evaluates **eight machine learning classifiers** to develop a cost-effective screening strategy based on survey data.

A key challenge in the dataset is the **83% / 17% class imbalance**, which is addressed using **SMOTE (Synthetic Minority Oversampling Technique)** applied only to the training data. This results in approximately a **49% increase in model sensitivity**, improving detection of at-risk individuals.

---

## Team Members
- **Vinamratha Raghavendra Jagirdar**(A20614061)*(Project Leader)*
- Naman Singh (A20574372)
- Urjita Saxena (A20578349)
- Deepali Budhiraja (A20615733)

---

## Key Features & Methodology

### Data Source
- **2015 CDC Diabetes Health Indicators Dataset**
- **229,787 records**

### Target Variable
- **Binary Classification**
  - `0` → No Diabetes  
  - `1` → Diabetes / Prediabetes  

### Data Preprocessing
- Data deduplication  
- Stratified **80/20 train-test split**  
- **SMOTE applied only to training data** (prevents data leakage)  
- Feature scaling and centering  

---

## Model Portfolio
- Logistic Regression  
- Linear Discriminant Analysis (LDA)  
- K-Nearest Neighbors (KNN)  
- Decision Trees  
- Random Forest  
- XGBoost  
- Ridge Regression  
- Lasso Regression  

---

## Advanced Analytics
- **Fairness Audit** across:
  - Income  
  - Age  
  - Sex  

- **Bootstrapped Confidence Intervals** for model accuracy  

- **Feature Interpretability**
  - SHAP (SHapley Additive exPlanations)  
  - Gini Importance  

---

## Performance Summary

| Model                | AUC-ROC | Accuracy | Sensitivity |
|---------------------|--------|----------|-------------|
| XGBoost             | 0.804  | —        | —           |
| Logistic Regression | 0.798  | ~0.74    | ~0.69       |
| LDA                 | 0.798  | ~0.74    | ~0.69       |
| Random Forest       | 0.780  | 0.869    | 0.289       |

### Key Insights
- **XGBoost** achieved the best discrimination (highest AUC-ROC).
- **Logistic Regression** provided the best balance between performance and interpretability.

---

## Requirements

### Programming Language
- R

### Libraries

**Data Manipulation**
- tidyverse  
- scales  

**Machine Learning**
- caret  
- randomForest  
- xgboost  
- smotefamily  
- glmnet  
- e1071  
- class  

**Visualization**
- corrplot  
- ggcorrplot  
- rpart.plot  
- gridExtra  
- vip  

**Evaluation**
- pROC  
- fastshap  
- boot  
- car  

**Compute**
- parallel  
- doParallel  

---

## How to Run

### 1. Load Data
Ensure `diabetes.csv` is in your working directory and change the path in 'CDC.R' correctly.

### 2. Parallel Setup
The script automatically detects available CPU cores to accelerate training.

### 3. Checkpoints
You can resume progress using saved files:
- `CDC_checkpoint1_models.RData` → After training  
- `CDC_checkpoint2_evaluated.RData` → After evaluation  

### 4. Output
All plots are exported to: CDC_all_plots.pdf
