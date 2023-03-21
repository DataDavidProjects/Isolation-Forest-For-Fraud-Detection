import sys
sys.path.append("/Isolation-Forest-For-Fraud-Detection/src/")
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from src.preprocessing.data import read_all_trx, train_test_split_transactions
from src.preprocessing.features import create_feature_matrix
from src.model.performance import evaluate_model, random_search_cv,time_window_cv
from src.model.anomalydetection import MahalanobisOneclassClassifier
from src.preprocessing.helpers import scenario_sample


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
X = create_feature_matrix(transactions_df,windows_size_in_days = [1,5,7,15],delay_period=7)
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
#___________________________________________________________________


# _____________________ HYPCV ______________________________________
contamination = [i/1000 for i in range(1,11)]
n_estimators = [i*10 for i in range(1,16)]
max_samples = [i*1000 for i in range(1,11)]
param_grid={ 'n_estimators':n_estimators,
             'contamination':contamination,
             "max_samples":max_samples,
             'bootstrap':[False,True],
             'n_jobs':[-1],
             'random_state':[None],
             'verbose':[0]}

y = X[target]
best_params, best_score = random_search_cv(IsolationForest(),param_grid,X[features],y)
model = IsolationForest(**best_params)
best_params_cv = {'n_estimators': 120, 'contamination': 0.008, 'max_samples': 10000,
                  'bootstrap': True, 'n_jobs': -1, 'random_state': None, 'verbose': 0}
#______________________________________________________________


#________________________ STANDARD INDEPENDENT SPLIT _________________________
X_train,X_test,y_train,y_test = train_test_split_transactions(X, features, train_start="2018-04-01",
                                                              train_end="2018-07-01", test_start="2018-07-01",
                                                              test_end="2018-09-01", target="TX_FRAUD")
# Fitting the model
model.fit(X_train)
# Reports Performance
benchmark = evaluate_model(model, X_test, y_test)
#____________________________________________________________________________




#________________________ SCENARIO  SPLIT ___________________________________
scenario_sensitivity = []
for scenario in transactions_df["TX_FRAUD_SCENARIO"].unique()[1:]:
    scenario_x = scenario_sample(transactions_df,scenario)
    X_train,X_test,y_train,y_test = train_test_split_transactions(create_feature_matrix(scenario_x,
                                                                                        windows_size_in_days=[1, 5, 7, 15],
                                                                                        delay_period=7),
                                                                  features,
                                                                  train_start="2018-04-01", train_end="2018-07-01",
                                                                  test_start="2018-08-01",  test_end="2018-09-01",
                                                                  target="TX_FRAUD")
    # Fitting the model
    model.fit(X_train[features])
    # Reports Performance
    benchmark = evaluate_model(model, X_test, y_test)
    scenario_sensitivity.append(benchmark)

scenario_sensitivity = pd.concat(scenario_sensitivity,axis=1).T
scenario_sensitivity.index = [f"scenario_{i}" for i in range(1,4)]
scenario_sensitivity
#____________________________________________________________________________



#______________________________ TIME WINDOW SPLIT____________________________
model = IsolationForest()
results_ts = time_window_cv(transactions_df,model,features)
#____________________________________________________________________________


#____________________________________________________________________________

condition = transactions_df["TX_DATETIME"].between(results_ts.loc[0,"test_start"], results_ts.loc[0,"test_end"])
transactions_df.loc[condition,"TX_FRAUD_SCENARIO"].value_counts()

condition = transactions_df["TX_DATETIME"].between(results_ts.loc[1,"test_start"], results_ts.loc[1,"test_end"])
transactions_df.loc[condition,"TX_FRAUD_SCENARIO"].value_counts()
#____________________________________________________________________________



#________________________ FEATURE INDEPENDENT SPLIT _________________________
basic = ['TX_AMOUNT','TX_TIME_SECONDS', 'TX_TIME_DAYS']
features = ['TX_AMOUNT'] + flag_features + terminal_features + customer_features + time_features
auc_are_feat_type  = pd.DataFrame()
for feats in [basic,customer_features,time_features,terminal_features]:
    # Reports Performance
    auc_are_feat_type = pd.concat([auc_are_feat_type,time_window_cv(transactions_df,model,feats)],0)

indexlist = []
for feat in ["raw_features", "customer_features", "time_features", "terminal_features"]:
    for time in range(1, 3):
        indexname = feat + f"_time_folder_{time}"
        indexlist.append(indexname)
auc_are_feat_type.index = indexlist

validation_values  = auc_are_feat_type["validation"].explode()[0::3]
validation_values.name = "AUC"
validation_values.index = indexlist
auc_are_feat_type = pd.concat([auc_are_feat_type,validation_values],1)
#____________________________________________________________________________
