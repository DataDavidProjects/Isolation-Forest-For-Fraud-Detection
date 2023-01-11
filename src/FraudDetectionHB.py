import pandas as pd
import numpy as np
import datetime
import sys
sys.path.append("/Isolation-Forest-For-Fraud-Detection/src/")
from src.preprocessing.data import read_data,read_all_trx
from src.preprocessing.features import create_feature_matrix
from src.preprocessing.helpers import TimeBasedCV
from src.model.validation import model_validation


calendar = pd.date_range('2018-04-01', '2018-09-30',  inclusive="both").strftime('%Y-%m-%d')
root = "https://github.com/Fraud-Detection-Handbook/simulated-data-raw/blob/main/data/"
path_data = [f"{root}{date}.pkl?raw=true" for date in calendar]


transactions_df = read_all_trx(path_data).sort_values('TX_DATETIME').reset_index(drop=True)
X = create_feature_matrix(transactions_df,windows_size_in_days = [1,5,7,15,30],delay_period=7)
target = "TX_FRAUD"
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

tscv = TimeBasedCV(train_period=30*3, test_period=30,freq='days')
index_output = tscv.split(X, validation_split_date=datetime.date(2018,9,1))
model = model_validation(X,target = "TX_FRAUD",cv = index_output)
results  = model.cv_results_

for train_index, test_index in tscv.split(X,validation_split_date=datetime.date(2018, 4, 1), date_column="TX_DATETIME"):
    print(train_index, test_index)



