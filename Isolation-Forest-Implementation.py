import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn import  metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
import time




##################### Reading Data #########################
mac = False
if mac:
    data_path = "data/transactions"
    df = pd.read_csv(data_path,index_col =0)
    df.drop(['merchantCity', 'merchantState', 'merchantZip', 'echoBuffer', 'posOnPremises', 'recurringAuthInd'],
            axis=1,
            inplace=True)
else :
    data_path = "data/transactions.txt"
    df = pd.read_json(data_path, lines=True)
    df.drop(['merchantCity', 'merchantState', 'merchantZip', 'echoBuffer', 'posOnPremises', 'recurringAuthInd'],
            axis=1,
            inplace=True)
#############################################################


##################### Data Matrix Definition ###############################
numeric_columns = ['availableMoney', 'creditLimit', 'currentBalance', 'transactionAmount']

categorical_columns = ['accountNumber', 'acqCountry', 'cardCVV', 'cardLast4Digits', 'cardPresent',
                       'customerId', 'dateOfLastAddressChange', 'enteredCVV','expirationDateKeyInMatch', 'merchantCategoryCode',
                       'merchantCountryCode', 'merchantName', 'posConditionCode', 'posEntryMode',
                       'transactionType']

date_columns = ['accountOpenDate', 'currentExpDate', 'transactionDateTime']
target = ["isFraud"]

data_matrix = df.loc[:,numeric_columns+categorical_columns+date_columns]
y = df.loc[:,"isFraud"].astype(int)
#############################################################


#################### Preprocessing ##########################
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


class DateEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, by=1, columns=None):
        self.by = by
        self.columns = columns

    def fit(self, X, y=None):
        columns = X.columns if self.columns is None else self.columns
        return self

    def transform(self,X, y=None):
        columns = X.columns if self.columns is None else self.columns
        output = X.copy()
        # Converting in datetime format
        time_frames_str = ["year", "month", "day", "hour", "minute", "second"]
        for col in columns:
            # cast as time
            output[col] = pd.to_datetime(output[col])
            for time in time_frames_str:
                try:
                    # create the new column with time frame information
                    output[col+"_"+time] = getattr(output[col].dt, time)
                except:
                    print(f"Problems in {col,time}")
                    pass
            # drop original columns
            output = output.drop(col,axis = 1)
        return output
#############################################################


##################### Pipeline ##############################
# Expected columns out of the preprocessing pipeline
encoded_data_columns = [col +"_"+ time for col in date_columns for time in ["year", "month", "day", "hour", "minute", "second"] ]
pipeline_out_columns = categorical_columns+encoded_data_columns+numeric_columns

# Init Pipeline
preprocessing_pipeline  = ColumnTransformer(
    [
        ("MultiColumnLabelEncoder",MultiColumnLabelEncoder(),categorical_columns),
        ("DataEncoder", DateEncoder(), date_columns),
    ],remainder="passthrough"
)


IF = IsolationForest(n_estimators=100, max_samples='auto',
                     max_features=1.0, bootstrap=False, n_jobs=-1, random_state=42, verbose=0)
complete_pipeline = Pipeline([
    ('Preprocessing', preprocessing_pipeline),
    ('Model',IF)
])
#############################################################


##################### Cross Validation ######################
# Need to perform a stratified K fold CV
# Use f1 metric because we want to see how good we have generalized frauds

#############################################################




##################### Results ##############################


############################################################

