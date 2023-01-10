import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import IsolationForest
from random import randint, uniform
import sys
sys.path.append("/Isolation-Forest-For-Fraud-Detection/src/")
from src.preprocessing.data import read_data,read_all_trx
from src.preprocessing.features import create_feature_matrix
from src.preprocessing.helpers import TimeBasedCV


calendar = pd.date_range('2018-04-01', '2018-09-30',  inclusive="both").strftime('%Y-%m-%d')
root = "https://github.com/Fraud-Detection-Handbook/simulated-data-raw/blob/main/data/"
path_data = [f"{root}{date}.pkl?raw=true" for date in calendar]


transactions_df = read_all_trx(path_data).sort_values('TX_DATETIME').reset_index(drop=True)


X = create_feature_matrix(transactions_df,windows_size_in_days = [1,5,7,15,30],delay_period=7)
target = "TX_FRAUD"


tscv = TimeBasedCV(train_period=30*3, test_period=30,freq='days')
index_output = tscv.split(X, validation_split_date=datetime.date(2019,2,1))

estimator = IsolationForest()
model_params = {'n_estimators':np.linspace(10,500, num=20, retstep=False).astype(int).tolist(),
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

model = RandomizedSearchCV(
                            estimator = estimator,
                            param_distributions = model_params,
                            n_iter = 10,
                            n_jobs = -1,
                            iid = True,
                            cv = index_output,
                            verbose=5,
                            pre_dispatch='2*n_jobs',
                            random_state = None,
                            return_train_score = True)

model.fit(X.drop(target, axis=1),X[target])
results = model.cv_results_
