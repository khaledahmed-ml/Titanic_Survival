# Titanic Survival Prediction

## Overview
Predicting passenger survival on the Titanic using Machine Learning classification models.

## Dataset
- Source: Kaggle Titanic Dataset ( Titanic_train.csv )
- 891 passengers, 12 features

## What I Did
1. Dropped irrelevant columns (PassengerId, Name, Cabin, etc.)
2. Filled missing Age values with mean
3. Encoded Sex column to numeric
4. Scaled features using StandardScaler
5. Trained Logistic Regression with L2 regularization
6. Evaluated using accuracy, classification report and confusion matrix
7. Built a custom prediction to test survival probability

## Models Used
- Logistic Regression (Titanic-Disaster-Log.py)
- SVM (Titanic-Disaster-SVM.py)

## Results
| Model               | Accuracy |
|---------------------|----------|
| Logistic Regression | ~80%     |
| SVM                 | ~82%     |

## Libraries
- Python
- Pandas
- Scikit-Learn
- Matplotlib
- Seaborn

## How to Run
1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Run: python titanic_logistic.py
4. Don't forget to change the file path!!


