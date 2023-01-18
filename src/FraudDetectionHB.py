import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import sys
sys.path.append("/Isolation-Forest-For-Fraud-Detection/src/")
from src.preprocessing.data import read_all_trx
from src.preprocessing.features import create_feature_matrix
from src.preprocessing.helpers import TimeBasedCV
from src.model.validation import tune_isolation_forest

start = '2018-04-01'
end = '2018-09-30'
calendar = pd.date_range(start, end,inclusive="both").strftime('%Y-%m-%d')
root = "https://github.com/Fraud-Detection-Handbook/simulated-data-raw/blob/main/data/"
path_data = [f"{root}{date}.pkl?raw=true" for date in calendar]
transactions_df = read_all_trx(path_data).sort_values('TX_DATETIME').reset_index(drop=True)

X = create_feature_matrix(transactions_df,windows_size_in_days = [1,5,7,15,30],delay_period=7)
target = "TX_FRAUD"
index = "TX_DATETIME"
features =['TX_AMOUNT','TX_DURING_WEEKEND', 'TX_DURING_NIGHT','CUSTOMER_ID_NB_TX_1DAY_WINDOW',
           'CUSTOMER_ID_AVG_AMOUNT_1DAY_WINDOW', 'CUSTOMER_ID_NB_TX_7DAY_WINDOW',
           'CUSTOMER_ID_AVG_AMOUNT_7DAY_WINDOW', 'CUSTOMER_ID_NB_TX_30DAY_WINDOW',
           'CUSTOMER_ID_AVG_AMOUNT_30DAY_WINDOW', 'TERMINAL_ID_NB_TX_1DAY_WINDOW',
           'TERMINAL_ID_RISK_1DAY_WINDOW', 'TERMINAL_ID_NB_TX_7DAY_WINDOW',
           'TERMINAL_ID_RISK_7DAY_WINDOW', 'TERMINAL_ID_NB_TX_30DAY_WINDOW',
           'TERMINAL_ID_RISK_30DAY_WINDOW']

train_period = "2018-08-01"
X_train = X.loc[X[index] < train_period][features]
X_test = X.loc[(X[index] >= train_period) & (X[index] < end)][features]

y_train = X.loc[X[index] < train_period][target]
y_test = X.loc[(X[index] >= train_period) & (X[index] < end)][target]

from sklearn.ensemble import IsolationForest
iso_Forest = IsolationForest(n_estimators=100, max_samples=1000, contamination=0.02, random_state=2018)
# Fitting the model
iso_Forest.fit(X_train[features])
X_test["scores"] = iso_Forest.score_samples(X_test[features])
alert = np.percentile(X_test["scores"].values,1)
X_test["anomaly"] = X_test["scores"] < alert
# AUC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,X_test.anomaly.astype(int))
# Ploting the graph to identify the anomolie score
plt.figure(figsize=(12, 8))
plt.hist(X_test["scores"], bins=50);



# _____________________ HYPCV ____________________________________
contamination = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
n_estimators = np.linspace(50,500, num=20, retstep=False).astype(int).tolist()
max_samples = ["auto"]+[i/1000 for i in range(1,11)]

param_grid={ 'n_estimators':n_estimators,
             'contamination':contamination,
             "max_samples":max_samples,
             'behaviour':['old','new'],
             'bootstrap':[False,True],
             'n_jobs':[-1],
             'random_state':[None],
             'verbose':[0]}

search = tune_isolation_forest(X_train.values, y_train.values)
best_estimator = search.best_estimator_
best_params = search.best_params_
#______________________________________________________________






# ____________________ CV RESULTS_________________________________
results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
#______________________________________________________________




