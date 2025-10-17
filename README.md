# Credit Card Default Prediction: Imputation Strategy Optimization
## Overview
This project conducts a systematic investigation of different imputation techniques in the context of credit risk modeling. The analysis evaluates how various strategies for handling missing data impact the performance of downstream credit default prediction models, providing evidence-based recommendations for optimizing imputation in financial risk assessment.

## Author
- **Name:** S SHRIPRASAD
- **Roll:** DA25E054

## Problem Statement
Missing data is a common challenge in real-world financial datasets. When building credit risk models, analysts must decide how to handle these gaps, but there's often little empirical evidence to guide this decision. This project addresses the question: **Which imputation strategy maximizes the predictive performance of credit default models?**

## Dataset
The analysis uses the UCI Credit Card dataset, which contains information about credit card clients and their payment history. The dataset includes demographic information (age, gender, education, marital status), credit data (bill and payment amounts), payment history, and default status.

## Methodology

### 1. Data Preparation & Simulation of Missingness
- Started with a complete dataset and artificially introduced 8% missing values into the `AGE` and `BILL_AMT1` columns to simulate a realistic problem
- Created multiple versions of the dataset, each featuring a different strategy for handling missing values

### 2. Imputation Strategies Implemented
- **Simple Statistical Fill**: Median imputation
- **Model-Based Imputation**:
  - Linear Regression
  - K-Nearest Neighbors (KNN)
  - Decision Tree
- **Advanced Technique**: Iterative Imputer (MICE)
- **Control Strategy**: Listwise Deletion (removing rows with missing values)

### 3. Downstream Model Training
- Trained a Logistic Regression classifier on each imputed dataset
- Used `class_weight='balanced'` to handle the imbalanced distribution of defaults
- Evaluated performance using F1-score and Recall metrics

### 4. Performance Analysis
- Compared F1-scores across different imputation strategies
- Analyzed Recall as a more business-relevant metric for credit risk modeling
- Investigated the impact of imputation on statistical relationships within the data
- Examined feature importance and correlations with the target variable

## Key Findings

1. **Initial F1-Score Deadlock**: All imputation strategies resulted in nearly identical F1-scores (difference between best and worst was just 0.0065), suggesting that imputation choice had minimal impact on overall classification balance.

2. **Recall as a Critical Metric**: When evaluating based on Recall (the ability to identify actual defaulters), clear differences emerged, with Median Imputation performing best among strategies that preserved all data points.

3. **The Low-Impact Feature Insight**: Correlation analysis revealed that `AGE` had near-zero correlation (0.02) with default status, explaining why sophisticated imputation techniques didn't yield significant improvements.

4. **Complexity vs. Performance Trade-off**: More complex imputation methods (Decision Tree, Iterative) introduced subtle noise into a low-impact feature, slightly degrading the performance of the simple linear classifier.

## Conclusion & Recommendation

The project recommends **Median Imputation** as the optimal strategy for this credit risk modeling task based on:

1. **Superior Performance**: Highest Recall among methods that retained the full dataset
2. **Principle of Parsimony**: For low-impact features, simpler methods that minimize disruption often outperform technically sophisticated approaches
3. **Balance of Performance and Safety**: Retains all customers in the dataset, avoiding potential biases from row deletion

## Key Lesson
The "best" imputation technique is always context-dependent. For features with strong predictive signals, sophisticated imputation models may be valuable. However, for low-impact features, the goal shifts from accuracy to minimizing disruption, where simpler methods often excel.

## Technical Implementation
The analysis was implemented in Python, utilizing:
- **Core libraries**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Machine Learning**: scikit-learn (imputation methods, preprocessing, and classification models)
