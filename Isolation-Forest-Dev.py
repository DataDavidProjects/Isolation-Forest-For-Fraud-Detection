import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn import  metrics
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

#https://www.kaggle.com/code/ranjeetshrivastav/fraud-detection-pycaret
#https://www.kaggle.com/datasets/kartik2112/fraud-detection
data_path = "data/transactions.txt"
# In mac probably csv
df = pd.read_json(data_path,lines=True)

le = LabelEncoder()
cat_columns = ['merchantName','acqCountry','merchantCountryCode','posEntryMode',
               'posConditionCode','merchantCategoryCode','transactionType',
               'cardPresent','expirationDateKeyInMatch','isFraud']
for i in cat_columns:
    df[i] = le.fit_transform(df[i])

# converting in datetime format
df['transactionDateTime'] = pd.to_datetime(df['transactionDateTime'])
df['currentExpDate'] = pd.to_datetime(df['currentExpDate'])
df['accountOpenDate'] = pd.to_datetime(df['accountOpenDate'])
df['dateOfLastAddressChange'] = pd.to_datetime(df['dateOfLastAddressChange'])

df['transactionDateTime_year'] = df['transactionDateTime'].dt.year
df['transactionDateTime_month'] = df['transactionDateTime'].dt.month
df['transactionDateTime_day'] = df['transactionDateTime'].dt.day
df['transactionDateTime_hour'] = df['transactionDateTime'].dt.hour
df['transactionDateTime_minute'] = df['transactionDateTime'].dt.minute
df['transactionDateTime_second'] = df['transactionDateTime'].dt.second

df['currentExpDate_year'] = df['currentExpDate'].dt.year
df['currentExpDate_month'] = df['currentExpDate'].dt.month
df['currentExpDate_day'] = df['currentExpDate'].dt.day

df['accountOpenDate_year'] = df['accountOpenDate'].dt.year
df['accountOpenDate_month'] = df['accountOpenDate'].dt.month
df['accountOpenDate_day'] = df['accountOpenDate'].dt.day

df['dateOfLastAddressChange_year'] = df['dateOfLastAddressChange'].dt.year
df['dateOfLastAddressChange_month'] = df['dateOfLastAddressChange'].dt.month
df['dateOfLastAddressChange_day'] = df['dateOfLastAddressChange'].dt.day

# drop datetime column
df.drop('transactionDateTime',axis = 1,inplace = True)
df.drop('currentExpDate',axis = 1,inplace = True)
df.drop('accountOpenDate',axis = 1,inplace = True)
df.drop('dateOfLastAddressChange',axis = 1,inplace = True)


#df.to_csv("data/transactions-preprocessed.txt",index=False)

############ DEV

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn import  metrics
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

raw = pd.read_json("data/transactions.txt",lines=True)
########### Pipeline ################

class MultiColumnLabelEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None):
        self.columns = columns # array of column names to encode

    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = LabelEncoder().fit(X[col])
        return self

    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[col])
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)

    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[col])
        return output


class Preprocessing(BaseEstimator, TransformerMixin):

    def __init__(self, by=1, columns=None):
        self.by = by
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self,X, y=None):
        # converting in datetime format
        X['transactionDateTime'] = pd.to_datetime(X['transactionDateTime'])
        X['currentExpDate'] = pd.to_datetime(X['currentExpDate'])
        X['accountOpenDate'] = pd.to_datetime(X['accountOpenDate'])
        X['dateOfLastAddressChange'] = pd.to_datetime(X['dateOfLastAddressChange'])

        X['transactionDateTime_year'] = X['transactionDateTime'].dt.year
        X['transactionDateTime_month'] = X['transactionDateTime'].dt.month
        X['transactionDateTime_day'] = X['transactionDateTime'].dt.day
        X['transactionDateTime_hour'] = X['transactionDateTime'].dt.hour
        X['transactionDateTime_minute'] = X['transactionDateTime'].dt.minute
        X['transactionDateTime_second'] = X['transactionDateTime'].dt.second

        X['currentExpDate_year'] = X['currentExpDate'].dt.year
        X['currentExpDate_month'] = X['currentExpDate'].dt.month
        X['currentExpDate_day'] = X['currentExpDate'].dt.day

        X['accountOpenDate_year'] = X['accountOpenDate'].dt.year
        X['accountOpenDate_month'] = X['accountOpenDate'].dt.month
        X['accountOpenDate_day'] = X['accountOpenDate'].dt.day

        X['dateOfLastAddressChange_year'] = X['dateOfLastAddressChange'].dt.year
        X['dateOfLastAddressChange_month'] = X['dateOfLastAddressChange'].dt.month
        X['dateOfLastAddressChange_day'] = X['dateOfLastAddressChange'].dt.day
        return X


# Final Pipeline
cat_columns =  ['merchantName','acqCountry','merchantCountryCode','posEntryMode',
               'posConditionCode','merchantCategoryCode','transactionType',
               'cardPresent','expirationDateKeyInMatch','isFraud']
pipeline  = ColumnTransformer(
    [
        ("preprocessing", Preprocessing(),raw.columns.tolist()),
        ("encoding",MultiColumnLabelEncoder(),cat_columns)
      ]
)

p_res  = pipeline.fit(raw).transform(raw)
p_res = pd.DataFrame(p_res)
print('Done!')