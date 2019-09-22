import pandas as pd
import quandl, math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


df = quandl.get('WIKI/GOOGL', api_key = "hWH3qS82LVUGJZ4-xBuq")
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df ['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df ['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-999999, inplace=True)

forecast_out = int(math.ceil(0.01*len(df)))
print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out:]


df.dropna(inplace=True)
y = np.array(df['label'])
y = np.array(df['label'])
# y = y.reshape(X.shape)

# X = X.reshape(1, 1)

arr = np.arange(0,11)

print(arr)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train)
# confidence = clf.score(X_test, y_test)
# forecast_set = clf.predict(X_lately)

