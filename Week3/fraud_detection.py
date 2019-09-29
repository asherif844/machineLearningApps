import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn as sk
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale

df = pd.read_csv(
    'mlappsPythonJS/Week3/dataset/creditcard.csv', low_memory=False)

print(df.head())

# identify the breakout of the classification
df.groupby('Class').size()

fraud = df.loc[df['Class'] == 1]
non_fraud = df.loc[df['Class'] == 0]

print(
    f'The number of Fraudulent cases are {len(fraud)} and the number of non-fraudulent cases are {len(non_fraud)}.')

# create a scatterplot
ax = fraud.plot.scatter(x='Amount', y='Class', color='Orange', label='Fraud')
non_fraud.plot.scatter(x='Amount', y='Class',
                       color='Blue', label='Non-Fraud', ax=ax)
plt.show()

# split data into target (y) and predictors (x)

x = df.iloc[:, :-1]
y = df['Class']

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

x2 = pd.DataFrame(x_scaled)
x2.columns = x.columns
x = x2 

print(x.shape, y.shape)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.35)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

logistic = LogisticRegression(C=1e5)
logistic.fit(X_train, y_train)

print(f'score = {logistic.score(X_test, y_test)}')

y_predicted = logistic.predict(X_test)

np.array(y_predicted)

df_test = pd.DataFrame({'y_predict':y_predicted, 'y_test':y_test})

print(confusion_matrix(y_test, y_predicted))
print(accuracy_score(y_test, y_predicted)*100)
print(precision_score(y_test, y_predicted)*100)
print(recall_score(y_test, y_predicted)*100)

# print the test results in a confusion matrix

data = confusion_matrix(y_test, y_predicted)
df_cm = pd.DataFrame(data, columns=np.unique(y_test), index = np.unique(y_test))
df_cm.index.name = 'Actual'
df_cm.columns.name = 'Predicted'
plt.figure(figsize = (10,7))
sns.set(font_scale=1.4)#for label size
sns.heatmap(df_cm, cmap="Blues", annot=True,annot_kws={"size": 16})# font size