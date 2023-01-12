import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB

df = pd.read_csv('hotel2.csv')
print(df.head())
# df.replace('Not_Canceled', 1, inplace=True)
# df.replace('Canceled', 0, inplace=True)
X = df.iloc[:, 6:8].values
print(X)
y = df.iloc[:, 11].values
print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
model = GaussianNB()
# model = BernoulliNB()
# model = ComplementNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
accuracy = accuracy_score(y_test, y_pred)*100
print(accuracy)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
