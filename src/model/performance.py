from sklearn.metrics import make_scorer,roc_auc_score
from src.preprocessing.helpers import timer_decorator
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np


def evaluate_model(model, X_test, y_test):
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
    random_scores = np.random.uniform(size=len(y_test))
    dummy_not = np.zeros_like(y_test)
    # Summarize in benchmark
    benchmark = {
       "model":  roc_auc_score(y_test, scores),
       "dummy":  roc_auc_score(y_test, dummy_not),
       "random": roc_auc_score(y_test, random_scores)
    }
    return pd.Series(benchmark)

@timer_decorator
def random_search_cv(estimator, param_grid, X, y, n_iter=10, cv=5):
    auc_score = make_scorer(roc_auc_score,needs_threshold=True)
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
            y_pred = -estimator.score_samples(X_test)
            score = auc_score(y_test, y_pred)
            scores.append(score)
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = current_params
    return best_params, best_score
