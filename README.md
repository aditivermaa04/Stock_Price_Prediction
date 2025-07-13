# 📈 Stock Price Movement Prediction using Machine Learning

This project implements a machine learning pipeline to predict stock price movement (buy/sell signals) based on historical OHLC (Open, High, Low, Close) data of Tesla stock (2010–2017). The model classifies whether the next day's closing price will be higher than the current day's close.

---

## 🚀 Project Overview

- **Goal**: Predict a binary signal indicating if buying the stock would be profitable the next day
- **Approach**:
  - Feature engineering on raw OHLC data
  - Classification using traditional ML models
  - Evaluation using ROC-AUC and confusion matrix

---

## 🧠 Features Engineered

- `open-close`: Difference between opening and closing price
- `low-high`: Difference between low and high price
- `is_quarter_end`: Binary feature indicating if the day is a quarter-end
- `target`: Binary target indicating if next day's closing price > current day's close

---

## 🛠️ Tools & Technologies

- **Language**: Python
- **Libraries**: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost
- **IDE**: VS Code

---

## 📊 Models Used

- Logistic Regression
- Support Vector Machine (SVM with polynomial kernel)
- XGBoost Classifier

---

## 📈 Results

- **Best model**: XGBoost
- **Validation ROC-AUC**: 57.3%
- Confusion matrix analysis used for final evaluation

---

## 🗂️ Project Structure

