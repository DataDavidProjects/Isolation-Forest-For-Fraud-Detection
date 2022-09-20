import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn import  metrics
from sklearn.ensemble import IsolationForest

data_path = "data/transactions.txt"
df = pd.read_json(data_path,lines=True)
df.drop(['merchantCity','merchantState','merchantZip','echoBuffer','posOnPremises','recurringAuthInd'],axis=1,inplace=True)

le = LabelEncoder()
var = ['merchantName','acqCountry','merchantCountryCode','posEntryMode',
       'posConditionCode','merchantCategoryCode','transactionType',
       'cardPresent','expirationDateKeyInMatch','isFraud']
for i in var:
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


df.to_csv("data/transactions-preprocessed.txt",index=None)

