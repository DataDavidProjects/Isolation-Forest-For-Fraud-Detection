import sys
sys.path.append("/Isolation-Forest-For-Fraud-Detection/src/")
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.preprocessing.data import read_all_trx , train_test_split_transactions
from src.preprocessing.features import create_feature_matrix
from src.model.performance import evaluate_model ,random_search_cv



#______________________________ DATA______________________________________
start = '2018-04-01'
end = '2018-09-30'
calendar = pd.date_range(start, end, inclusive="both").strftime('%Y-%m-%d')
root = "https://github.com/Fraud-Detection-Handbook/simulated-data-raw/blob/main/data/"
path_data = [f"{root}{date}.pkl?raw=true" for date in calendar]
transactions_df = read_all_trx(path_data).sort_values('TX_DATETIME').reset_index(drop=True)
scenario_1 = transactions_df.loc[transactions_df["TX_FRAUD_SCENARIO"] == 1]
scenario_2 = transactions_df.loc[transactions_df["TX_FRAUD_SCENARIO"] == 2]
scenario_3 = transactions_df.loc[transactions_df["TX_FRAUD_SCENARIO"] == 3]
#_________________________________________________________________________


#__________________________ FEATURES _________________________________
X = create_feature_matrix(transactions_df,windows_size_in_days = [1,5,7,15,30],delay_period=7)
target = "TX_FRAUD"

index = "TX_DATETIME"

train_period = "2018-07-01"

customer_features = [i for i in X.columns if "CUSTOMER_ID_" in i]

flag_features = [i for i in X.columns if "TX_FLAG_" in i]

terminal_features = [i for i in X.columns if "TERMINAL_ID_" in i]

time_features = ['TX_TIME_SECONDS', 'TX_TIME_DAYS', 'TX_MONTH', 'TX_DAY', 'TX_HOUR', 'TX_MINUTE', 'TX_DURING_WEEKEND',
                 'TX_DURING_NIGHT']

helper_columns = ['TX_FRAUD', 'TX_FRAUD_SCENARIO', 'TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID']

features = ['TX_AMOUNT'] + flag_features + terminal_features + customer_features + time_features
#______________________________________________________________



# _____________________ HYPCV ______________________________________
contamination = [i/1000 for i in range(1,11)]
n_estimators = [i*10 for i in range(1,16)]
max_samples = [i*1000 for i in range(1,11)]
param_grid={ 'n_estimators':n_estimators,
             'contamination':contamination,
             "max_samples":max_samples,
             'behaviour':['old','new'],
             'bootstrap':[False,True],
             'n_jobs':[-1],
             'random_state':[None],
             'verbose':[0]}

y = X[target]
best_params, best_score = random_search_cv(IsolationForest(),param_grid,X[features],y)
#______________________________________________________________



#________________________ SPLIT _______________________________
X_train,X_test,y_train,y_test = train_test_split_transactions(X, features, train_start="2018-04-01",
                                                              train_end="2018-07-01", test_start="2018-07-01",
                                                              test_end="2018-09-01", target="TX_FRAUD")
#______________________________________________________________

#___________________________ MODEL________________________________
model = IsolationForest.set_params(**best_params)
# Fitting the model
model.fit(X_train)
# Reports Performance
report, cm, benchmark = evaluate_model(model, X_test, y_test)
#___________________________________________________________________







