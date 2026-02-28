# ❤️ AI Powered Heart Risk Prediction System

An intelligent Machine Learning web application that predicts the risk of heart disease using clinical parameters.

Developed using Logistic Regression and deployed with Streamlit.

---

## 🚀 Live Demo
https://heartstrockprediction.streamlit.app/

---

## 📌 Project Overview

This project uses a trained Logistic Regression model to predict whether a person is at **High Risk** or **Low Risk** of heart disease based on medical attributes such as:

- Age
- Sex
- Chest Pain Type
- Blood Pressure
- Cholesterol
- Fasting Blood Sugar
- Max Heart Rate
- Exercise Induced Angina
- ST Depression (Oldpeak)
- ST Slope

The model provides:
- Risk Classification (High / Low)
- Probability Percentage
- Confidence Score
- Health Recommendations

---

## 🧠 Machine Learning Details

- Algorithm: Logistic Regression
- Feature Scaling: StandardScaler
- Encoding: One-Hot Encoding
- Dataset: UCI Heart Disease Dataset
- Accuracy: ~85-90%

---

## 🛠 Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit
- Joblib
- Git & GitHub

---

## 📂 Project Structure
heart-risk-predictor/
│
├── app.py
├── Logistic_reg.pkl
├── scaler.pkl
├── columns.pkl
|__ requirements.txt
└── README.md
