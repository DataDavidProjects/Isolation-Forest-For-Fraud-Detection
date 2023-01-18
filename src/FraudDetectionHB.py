import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import sys
sys.path.append("/Isolation-Forest-For-Fraud-Detection/src/")
from src.preprocessing.data import read_all_trx , train_test_split_transactions
from src.preprocessing.features import create_feature_matrix
from src.model.performance import tune_isolation_forest

start = '2018-04-01'
end = '2018-09-30'
calendar = pd.date_range(start, end, inclusive="both").strftime('%Y-%m-%d')
root = "https://github.com/Fraud-Detection-Handbook/simulated-data-raw/blob/main/data/"
path_data = [f"{root}{date}.pkl?raw=true" for date in calendar]
transactions_df = read_all_trx(path_data).sort_values('TX_DATETIME').reset_index(drop=True)

scenario_1 = transactions_df.loc[transactions_df["TX_FRAUD_SCENARIO"] == 1]
scenario_2 = transactions_df.loc[transactions_df["TX_FRAUD_SCENARIO"] == 2]
scenario_3 = transactions_df.loc[transactions_df["TX_FRAUD_SCENARIO"] == 3]



X = create_feature_matrix(transactions_df,windows_size_in_days = [1,5,7,15,30],delay_period=7)
target = "TX_FRAUD"
index = "TX_DATETIME"
features =[]

train_period = "2018-07-01"
X_train,X_test,y_train,y_test = train_test_split_transactions(X)


iso_Forest = IsolationForest(n_estimators=100, max_samples=2000, contamination=0.002, random_state=2018)
# Fitting the model
iso_Forest.fit(X_train)

# AUC
from sklearn.metrics import roc_auc_score ,classification_report
X_test["scores"] = iso_Forest.score_samples(X_test)
alert = np.percentile(X_test["scores"].values,1)
X_test["anomaly"] = X_test["scores"] < alert
print(roc_auc_score(y_test,X_test.anomaly.astype(int)))
print(classification_report(y_test,X_test.anomaly.astype(int)))

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




