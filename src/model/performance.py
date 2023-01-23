import pandas as pd
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.preprocessing.helpers import timer_decorator
from src.preprocessing.features import create_feature_matrix
from src.preprocessing.data import train_test_split_transactions


def evaluate_model(model, X_test, y_test,benchmark=True):
    """
        Evaluates the performance of the model using AUC

        Parameters:
        model (object): The trained model that will be used to make predictions
        X_test (pd.DataFrame): The test set
        y_test (pd.Series): The true labels of the test set

        Returns:
        pd.DataFrame: A dataframe containing  benchmark
    """
    # Make predictions using model,random and dummy
    # Note that model score are flipped for scikit learn evaluation greater better
    scores = -model.score_samples(X_test)
    if benchmark:
        random_scores = np.random.uniform(size=len(y_test))
        dummy_not = np.zeros_like(y_test)
        # Summarize in benchmark
        benchmark = {
            "model": roc_auc_score(y_test, scores),
            "dummy": roc_auc_score(y_test, dummy_not),
            "random": roc_auc_score(y_test, random_scores)
        }
        result = pd.Series(benchmark)
    else:
        result = roc_auc_score(y_test, scores)
    return result

@timer_decorator
def random_search_cv(estimator, param_grid, X, y, n_iter=10, cv=5):
    """
           Evaluates the performance of the model using RandomizedSearchCV

           Parameters:
           estimator (object): The trained model that will be used to make predictions
           X (pd.DataFrame): The data set
           y (pd.Series): The true labels of the data set
           param_grid (dict): Parameter of estimator

           Returns:
           best_score: A variable containing  benchmark
           best_params: A variable containing  params
       """
    skf = StratifiedKFold(n_splits=cv, random_state=None,shuffle=False)
    best_score = 0
    best_params = {}
    for i in range(n_iter):
        current_params = {k: np.random.choice(v) for k, v in param_grid.items()}
        scores = []
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            estimator.set_params(**current_params)
            estimator.fit(X_train)
            # Note flipped scores for IsolationForest
            anomaly_scores = -estimator.score_samples(X_test)
            score = roc_auc_score(y_test, anomaly_scores)
            scores.append(score)
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = current_params
    return best_params, best_score


from datetime import datetime as dt
def time_window_cv(X,model,features,target="TX_FRAUD", months_in_train_window = 2,months_in_test_window = 1):
    min_period = X["TX_DATETIME"].min()
    max_period = X["TX_DATETIME"].max()
    train_start = min_period
    results = []
    print("Creating features...")
    X = create_feature_matrix(X, windows_size_in_days=[1, 5, 7, 15, 30], delay_period=7)
    print("Start Time Cross Validation...")
    while True:
        train_end = train_start + pd.DateOffset(months=months_in_train_window)
        test_start = train_end + pd.DateOffset(months=months_in_test_window)
        test_end = test_start + pd.DateOffset(months=months_in_test_window)
        if test_end > max_period:
            break
        print("Train:", train_start, train_end)
        print("Test:", test_start, test_end)
        X_train, X_test, y_train, y_test = train_test_split_transactions(X, features,
                                                                         train_start=train_start,target=target)
        start_time = time.time()
        model.fit(X_train[features])
        end_time = time.time()
        execution_time = end_time - start_time
        validation = evaluate_model(model, X_test, y_test)
        results.append(
            {'train_start': train_start, 'train_end': train_end, 'test_start': test_start, 'test_end': test_end,
             'execution_time': execution_time, 'model': model, 'validation': validation})
        train_start += pd.DateOffset(months=1)
    return pd.DataFrame(results)

