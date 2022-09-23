import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from datetime import datetime as dt
from dateutil.relativedelta import *
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn import  metrics
from sklearn.model_selection import  StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
import time




##################### Reading Data #########################
mac = False
if mac:
    data_path = "data/transactions"
    df = pd.read_csv(data_path,index_col =0)
    df["transactionDateTime"] = pd.to_datetime(df["transactionDateTime"])
    df.drop(['merchantCity', 'merchantState', 'merchantZip', 'echoBuffer', 'posOnPremises', 'recurringAuthInd'],
            axis=1,
            inplace=True)
    df = df.sort_values(by="transactionDateTime")
else :
    data_path = "data/transactions.txt"
    df = pd.read_json(data_path, lines=True)
    df["transactionDateTime"] = pd.to_datetime(df["transactionDateTime"])
    df.drop(['merchantCity', 'merchantState', 'merchantZip', 'echoBuffer', 'posOnPremises', 'recurringAuthInd'],
            axis=1,
            inplace=True)
    df = df.sort_values(by="transactionDateTime")
#############################################################


##################### Data Matrix Definition ###############################
numeric_columns = ['availableMoney', 'creditLimit', 'currentBalance', 'transactionAmount']

categorical_columns = ['accountNumber', 'acqCountry', 'cardCVV', 'cardLast4Digits', 'cardPresent',
                       'customerId', 'dateOfLastAddressChange', 'enteredCVV','expirationDateKeyInMatch',
                       'merchantCategoryCode','merchantCountryCode', 'merchantName', 'posConditionCode',
                       'posEntryMode','transactionType']

date_columns = ['accountOpenDate', 'currentExpDate', 'transactionDateTime']
target = ["isFraud"]

X = df.loc[:,numeric_columns+categorical_columns+date_columns]
y = df.loc[:,"isFraud"].astype(int)
#############################################################


#################### Preprocessing ##########################
class MultiColumnCategoricalEncoder(BaseEstimator, TransformerMixin):

    """"
    Returns
    ------------
            for each catogorical columns:
                    LabelEncoder fitted and transformed independently

    """

    def __init__(self, columns=None):
        self.columns = columns # array of column names to encode

    def fit(self, X, y=None):
        self.encoders = {}
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            self.encoders[col] = OrdinalEncoder( handle_unknown = 'use_encoded_value',unknown_value = -999).fit(X[[col]])
        return self

    def transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].transform(X[[col]])
        return output

    def fit_transform(self, X, y=None):
        return self.fit(X,y).transform(X)

    def inverse_transform(self, X):
        output = X.copy()
        columns = X.columns if self.columns is None else self.columns
        for col in columns:
            output[col] = self.encoders[col].inverse_transform(X[[col]])
        return output


class DateEncoder(BaseEstimator, TransformerMixin):
    '''
     Parameters-Fixed
    ----------
        time frame dates  ["year", "month", "day", "hour", "minute", "second"]
    Returns
    -------
    for each date column:
            year , month , day ,hour , minute, seconds in seperate columns

    '''

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


class TimeBasedCV(object):
    '''
    Parameters
    ----------
    train_period: int
        number of time units to include in each train set
        default is 30
    test_period: int
        number of time units to include in each test set
        default is 7
    freq: string
        frequency of input parameters. possible values are: days, months, years, weeks, hours, minutes, seconds
        possible values designed to be used by dateutil.relativedelta class
        deafault is days
    '''

    def __init__(self, train_period=30, test_period=7, freq='days'):
        self.train_period = train_period
        self.test_period = test_period
        self.freq = freq

    def split(self, data, validation_split_date=None, date_column='record_date', gap=0):
        '''
        Generate indices to split data into training and test set

        Parameters
        ----------
        data: pandas DataFrame
            your data, contain one column for the record date
        validation_split_date: datetime.date()
            first date to perform the splitting on.
            if not provided will set to be the minimum date in the data after the first training set
        date_column: string, deafult='record_date'
            date of each record
        gap: int, default=0
            for cases the test set does not come right after the train set,
            *gap* days are left between train and test sets

        Returns
        -------
        train_index ,test_index:
            list of tuples (train index, test index) similar to sklearn model selection
        '''

        # check that date_column exist in the data:
        try:
            data[date_column]
        except:
            raise KeyError(date_column)

        train_indices_list = []
        test_indices_list = []

        if validation_split_date == None:
            validation_split_date = data[date_column].min().date() + eval(
                'relativedelta(' + self.freq + '=self.train_period)')

        start_train = validation_split_date - eval('relativedelta(' + self.freq + '=self.train_period)')
        end_train = start_train + eval('relativedelta(' + self.freq + '=self.train_period)')
        start_test = end_train + eval('relativedelta(' + self.freq + '=gap)')
        end_test = start_test + eval('relativedelta(' + self.freq + '=self.test_period)')

        while end_test < data[date_column].max().date():
            # train indices:
            cur_train_indices = list(data[(data[date_column].dt.date >= start_train) &
                                          (data[date_column].dt.date < end_train)].index)

            # test indices:
            cur_test_indices = list(data[(data[date_column].dt.date >= start_test) &
                                         (data[date_column].dt.date < end_test)].index)

            print("Train period:", start_train, "-", end_train, ", Test period", start_test, "-", end_test,
                  "# train records", len(cur_train_indices), ", # test records", len(cur_test_indices))

            train_indices_list.append(cur_train_indices)
            test_indices_list.append(cur_test_indices)

            # update dates:
            start_train = start_train + eval('relativedelta(' + self.freq + '=self.test_period)')
            end_train = start_train + eval('relativedelta(' + self.freq + '=self.train_period)')
            start_test = end_train + eval('relativedelta(' + self.freq + '=gap)')
            end_test = start_test + eval('relativedelta(' + self.freq + '=self.test_period)')

        # mimic sklearn output
        index_output = [(train, test) for train, test in zip(train_indices_list, test_indices_list)]

        self.n_splits = len(index_output)

        return index_output

    def get_n_splits(self):
        """Returns the number of splitting iterations in the cross-validator
        Returns
        -------
        n_splits : int
            Returns the number of splitting iterations in the cross-validator.
        """
        return self.n_splits

#############################################################


##################### Pipeline ##############################
# Expected columns out of the preprocessing pipeline
time_frames = ["year", "month", "day", "hour", "minute", "second"]
encoded_data_columns = [col +"_"+ time for col in date_columns for time in time_frames]
pipeline_out_columns = categorical_columns+encoded_data_columns+numeric_columns

# Init Pipeline
preprocessing_pipeline  = ColumnTransformer(
    [
        ("TfidfVectorizer",MultiColumnCategoricalEncoder(),categorical_columns),
        ("DataEncoder", DateEncoder(), date_columns),
    ],
    remainder="passthrough"
)


IF = IsolationForest(n_estimators=100, max_samples='auto',
                     max_features=1.0, bootstrap=True,
                     n_jobs=-1, random_state=42, verbose=0)
complete_pipeline = Pipeline([
    ('Preprocessing', preprocessing_pipeline),
    ('Model',IF)
])
#############################################################


##################### Cross Validation ######################
tscv = TimeBasedCV(train_period=60,
                   test_period=10,
                   freq='days')
splits = tscv.split(X,
                    validation_split_date=datetime.date(2016,2,1),
                    date_column='transactionDateTime')
result_list = []
train_index_list = []
test_index_list = []
fitted_list = []

dev = True
if dev:
    import warnings
    warnings.simplefilter(action='ignore', category=FutureWarning)

for n,(train_index,test_index) in enumerate(splits):
    start = time.time()
    # Prepare Train Test
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    # Run Pipeline and Score
    fitted_pipeline = complete_pipeline.fit(X_train)
    anomaly_score = fitted_pipeline.decision_function(X_test)
    # Set the alert thresshold for cut
    alert = np.percentile(anomaly_score,30)


    predictions = [ 1  if i < alert else 0 for i in anomaly_score ]
    score = f1_score(y_test, predictions)
    # Save Score and params
    result_list.append(score)
    train_index_list.append(train_index)
    test_index_list.append(test_index)
    fitted_list.append(fitted_pipeline)
    # Running Time
    end = time.time()
    running_time = end - start
    print("_" * 30)
    print(f'Iteration {n} completed in {round(running_time, 3)} seconds, F1-score: {score}')
    print(metrics.classification_report(y_test, predictions))
    print("Proportions in train:")
    print(y_train.value_counts())


CV_results = pd.DataFrame(zip(result_list,train_index_list,test_index_list,fitted_list),
                          columns=["F1-score","train-idx","test-idx","fitted-pipeline"])
print(CV_results)
#############################################################




##################### Results ##############################


############################################################