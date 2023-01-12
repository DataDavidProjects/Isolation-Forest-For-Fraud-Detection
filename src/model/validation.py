from sklearn.ensemble import IsolationForest
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV
from src.preprocessing.helpers import timer_decorator

@timer_decorator
def tune_isolation_forest(X_train, y_train, param_grid=None, n_iter=10, cv=5, random_state=42):
    """
    Tune the hyperparameters of an Isolation Forest model using RandomizedSearchCV.

    Parameters:
        X_train (pd.DataFrame): Training set features
        y_train (pd.Series): Training set labels
        param_grid (dict, optional): Dictionary containing the parameter grid to use for the search. If not provided, a default grid will be used.
        n_iter (int, optional): Number of iterations for the RandomizedSearchCV.
        cv (int, optional): Number of folds for cross-validation.
        random_state (int, optional): Seed for the random number generator.

    Returns:
        best_estimator (IsolationForest): The best estimator found by the search.
        best_params (dict): The best hyperparameters found by the search.
    """
    if param_grid is None:
        param_grid = {'n_estimators': [100, 250, 500, 750, 1000],
                      'max_samples': [256, 512, 1024, 2048, 4096],
                      'contamination': [0.01, 0.05, 0.1, 0.15, 0.2],
                      'max_features': [1, 0.5, 'auto']}

    # Define the scoring metric

    scoring = make_scorer(roc_auc_score, greater_is_better=True, needs_threshold=True)

    # Create the Isolation Forest model
    model = IsolationForest(random_state=random_state)

    # Create the randomized search object
    search = RandomizedSearchCV(estimator=model, param_distributions=param_grid,
                                n_iter=n_iter, cv=cv, scoring=scoring, random_state=random_state,
                                n_jobs=-1, verbose=0)

    # Fit the search to the data
    search.fit(X_train, y_train)

    return search

#index_output = TimeBasedCV(train_period=30*3, test_period=30,freq='days').split(X, validation_split_date=datetime.date(2018,4,1),date_column="TX_DATETIME")
#results  = model.cv_results_
