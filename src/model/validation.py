import numpy as np
import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import IsolationForest


def model_validation(X,model_params,cv,target = "TX_FRAUD",estimator = IsolationForest()):
    """
    Run a model validation using K-fold cross validation technique with random search to find the best hyperparameters.

    Parameters:
    X : pd.DataFrame : The feature matrix
    model_params : Dict : Dictionnary of possible hyperparameters for the model
    cv : int : The number of fold used for K-fold cross validation
    target : str : the column name of the target variable
    estimator : sklearn estimator : The estimator object, (default is IsolationForest)

    Returns:
    sklearn.model_selection._search.RandomizedSearchCV : the object containing the fit model, the best hyperparamters and the evaluation scores
    """
    estimator = estimator
    model_params = model_params
    model = RandomizedSearchCV(
                                estimator = estimator,
                                param_distributions = model_params,
                                n_iter = 10,
                                n_jobs = -1,
                                cv = cv,
                                verbose=5,
                                random_state = None,
                                return_train_score = True)

    results = model.fit(X.drop(target, axis=1),X[target])

    return results