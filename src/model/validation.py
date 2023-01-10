import numpy as np
import datetime
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import IsolationForest

def model_validation(X,model_params,cv,target = "TX_FRAUD",estimator = IsolationForest()):

    estimator = estimator
    model_params = model_params
    model = RandomizedSearchCV(
                                estimator = estimator,
                                param_distributions = model_params,
                                n_iter = 10,
                                n_jobs = -1,
                                iid = True,
                                cv = cv,
                                verbose=5,
                                pre_dispatch='2*n_jobs',
                                random_state = None,
                                return_train_score = True)

    results = model.fit(X.drop(target, axis=1),X[target])

    return results