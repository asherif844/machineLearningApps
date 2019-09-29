import datetime
import math

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pandas_datareader.data as web
import seaborn as sns
from matplotlib import style
from pandas import DataFrame, Series
# Scale the X so that everyone can have the same distribution for linear regression
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

matplotlib.rc('figure', figsize=(8, 7))
style.use('ggplot')

start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2019, 1, 11)

df = web.DataReader("MSFT", 'yahoo', start, end)
print(df.tail())
print(df.head())

# df.loc(['Adj Close']).plot()
close_px = df['Adj Close']
moving_avg = close_px.rolling(window=100).mean()

df['moving_avg'] = moving_avg

moving_avg.plot(label='Moving Avg')
close_px.plot(label='MSFT')
plt.legend()
plt.show()

returns = close_px / close_px.shift(1) - 1

returns.plot(label='return')

dffreq = df.loc[:, ['Adj Close', 'Volume']]
dffreq['HL_%'] = (df['High'] - df['Low'])/df['Close']*100
dffreq['%_Change'] = (df['High'] - df['Low'])/df['Low']*100
dffreq.head(100)

# Drop missing value
dffreq.fillna(value=-99999, inplace=True)
# We want to separate 1 percent of the data to forecast
forecast_out = int(math.ceil(0.01 * len(dffreq)))
# Separating the label here, we want to predict the AdjClose
forecast_col = 'Adj Close'
dffreq['label'] = dffreq[forecast_col].shift(-forecast_out)
X = np.array(dffreq.drop(['label'], 1))

X = preprocessing.scale(X)
# Finally We want to find Data Series of late X and early X (train) for model generation and evaluation
X_lately = X[-forecast_out:]
X = X[:-forecast_out]
# Separate label and identify it as y
y = np.array(dffreq['label'])
y = y[:-forecast_out]

print(y.shape, X.shape)

X_train = X[:-23]
y_train = y

# Linear regression
clfreg = LinearRegression(n_jobs=-1)
clfreg.fit(X_train, y_train)
# Quadratic Regression 2
clfpoly2 = make_pipeline(PolynomialFeatures(2), Ridge())
clfpoly2.fit(X_train, y_train)

# Quadratic Regression 3
clfpoly3 = make_pipeline(PolynomialFeatures(3), Ridge())
clfpoly3.fit(X_train, y_train)


# KNN Regression
clfknn = KNeighborsRegressor(n_neighbors=2)
clfknn.fit(X_train, y_train)

X_train.shape
y_train.shape

X_test.shape
y_test.shape

X_test = X[-forecast_out:]
y_test = y[-forecast_out:]

confidencereg = clfreg.score(X_test, y_test)
confidencepoly2 = clfpoly2.score(X_test,y_test)
confidencepoly3 = clfpoly3.score(X_test, y_test)
confidenceknn = clfknn.score(X_test, y_test)

print(f'The linear regression confidence is {confidencereg}')
print(f'The quadratic regression 2 confidence is {confidencepoly2}')
print(f'The quadratic regression 3 confidence is {confidencepoly3}')
print(f'The knn regression confidence is {confidenceknn}')