import pandas as pd
import numpy as np
import datetime
import sys
sys.path.append("/Isolation-Forest-For-Fraud-Detection/src/")
from src.preprocessing.data import read_all_trx
from src.preprocessing.features import create_feature_matrix
from src.preprocessing.helpers import TimeBasedCV
from src.model.validation import tune_isolation_forest


calendar = pd.date_range('2018-04-01', '2018-09-30',  inclusive="both").strftime('%Y-%m-%d')
root = "https://github.com/Fraud-Detection-Handbook/simulated-data-raw/blob/main/data/"
path_data = [f"{root}{date}.pkl?raw=true" for date in calendar]

transactions_df = read_all_trx(path_data).sort_values('TX_DATETIME').reset_index(drop=True)
X = create_feature_matrix(transactions_df,windows_size_in_days = [1,5,7,15,30],delay_period=7)
target = "TX_FRAUD"
index = "TX_DATETIME"

param_grid={ 'n_estimators':np.linspace(10,500, num=20, retstep=False).astype(int).tolist(),
             'contamination':np.linspace(0.01,0.5, num=10, retstep=False).tolist()+['legacy'],
             'max_depth': np.linspace(10,30, num=10, retstep=False).astype(int).tolist(),
             'behaviour':['old','new'],
             'bootstrap':[True],
             'max_samples': ['auto'],
             'max_features': [1],
             'n_jobs':[-1],
             'random_state':[None],
             'verbose':[0],
             'warm_start':[True]}


features = ['TX_AMOUNT', 'TX_TIME_SECONDS', 'TX_TIME_DAYS']
X_train = X.loc[X[index] < "2018-06-01"][features]
y_train = X.loc[X[index] < "2018-06-01"][target]
X_test = X.loc[(X[index] >= "2018-06-01") & (X[index] < "2018-09-01")][features]
y_test = X.loc[(X[index] >= "2018-06-01") & (X[index] < "2018-09-01")][target]

best_estimator, best_params =tune_isolation_forest(X_train, y_train)


# * KPI on time behaviour