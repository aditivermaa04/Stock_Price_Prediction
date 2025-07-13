
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')




import os

current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, 'Tesla.csv')

print("Looking for file at:", file_path)  # Debug print

df = pd.read_csv(r'C:\Users\aditi\OneDrive\Desktop\Academic\get a job that mogs\Tesla.csv')

print(df.head())

print(df.info())
print(df.describe())


if (df['Close'] == df['Adj Close']).all():
    df.drop('Adj Close', axis=1, inplace=True)

    print(df.head())



print(df.isnull().sum())


plt.figure(figsize=(12,5))
plt.plot(df['Close'])
plt.title('Tesla Closing Price')
plt.ylabel('Price ($)')
plt.xlabel('Days')
plt.grid(True)
plt.show()


import seaborn as sb

features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.figure(figsize=(15,10))

for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.distplot(df[col])
    plt.title(f'Distribution of {col}')

plt.tight_layout()
plt.show()


plt.figure(figsize=(15,10))

for i, col in enumerate(features):
    plt.subplot(2, 3, i+1)
    sb.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


split_date = df['Date'].str.split('/', expand=True)
df['month'] = split_date[0].astype(int)
df['day'] = split_date[1].astype(int)
df['year'] = split_date[2].astype(int)


df['is_quarter_end'] = np.where(df['month'] % 3 == 0, 1, 0)

df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

df.dropna(inplace=True)



plt.pie(df['target'].value_counts(), labels=['Sell', 'Buy'], autopct='%1.1f%%')
plt.title("Target Distribution")
plt.show()


plt.figure(figsize=(10, 8))
sb.heatmap(df.drop('Date', axis=1).corr() > 0.9, annot=True, cbar=False)
plt.title("High Correlation Heatmap")
plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X = df[['open-close', 'low-high', 'is_quarter_end']]
y = df['target']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.1, random_state=2022)


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

models = [
    LogisticRegression(),
    SVC(kernel='poly', probability=True),
    XGBClassifier(use_label_encoder=False, eval_metric='logloss')
]

for model in models:
    model.fit(X_train, y_train)
    print(f"{model.__class__.__name__}")
    print("Train ROC AUC:", roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
    print("Validation ROC AUC:", roc_auc_score(y_valid, model.predict_proba(X_valid)[:, 1]))
    print("-" * 40)



from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(models[0], X_valid, y_valid)
plt.title("Confusion Matrix - Logistic Regression")
plt.show()






