# Fraud Detection System using Machine Learning

## 📌 Overview
This project implements an end-to-end fraud detection system using machine learning. It covers data exploration, handling class imbalance, model comparison, threshold optimization, and deployment using a Flask API.

The goal is to detect fraudulent credit card transactions while balancing fraud detection performance and customer experience.

---

## 🚀 Features
- Exploratory Data Analysis (EDA)
- Handling class imbalance using cost-sensitive learning
- Threshold tuning using Precision-Recall Curve
- Model comparison (Logistic Regression vs Random Forest)
- REST API for real-time fraud prediction
- Deployment-ready architecture

---

## 📊 Dataset
Credit Card Transactions Dataset (Highly imbalanced dataset)

> Note: Raw dataset is excluded from this repository
Fraud_Detection/
│
├── train.py # Training pipeline
├── app.py # Prediction API
├── notebooks/ # Experiments & analysis
├── data/ # Local dataset (ignored)
├── .gitignore
└── requirements.txt

📈 Business Impact

Improved fraud recall using class imbalance handling

Balanced fraud detection and customer friction using threshold optimization

Built a real-time scoring system for operational use

📌 Future Improvements

Deploy on cloud (Render / AWS / GCP)

Add monitoring and logging

Implement model retraining pipeline

Add authentication to API

👤 Author

Manoj Dhiman
