from sklearn.metrics import confusion_matrix,make_scorer,roc_auc_score,classification_report
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import RandomizedSearchCV
from src.preprocessing.helpers import timer_decorator


def cutoff_predict(model, X_test, threshold):
    """
      Returns binary predictions based on the anomaly score threshold

      Parameters:
      model (object): The trained model that will be used to make predictions
      X_test (pd.DataFrame): The test set
      threshold (float): Percentile threshold to use to make predictions

      Returns:
      list: List of binary predictions
    """
    # get the anomaly scores for the test set
    scores = model.decision_function(X_test)
    # calculate the threshold based on the percentile
    threshold = np.percentile(scores, threshold)
    # make binary predictions based on the threshold
    y_pred = [1 if s < threshold else 0 for s in scores]
    return y_pred

def evaluate_model(model, X_test, y_test):
    """
        Evaluates the performance of the model using AUC and classification report

        Parameters:
        model (object): The trained model that will be used to make predictions
        X_test (pd.DataFrame): The test set
        y_test (pd.Series): The true labels of the test set

        Returns:
        pd.DataFrame: A dataframe containing the classification report
    """
    y_pred = cutoff_predict(model, X_test, threshold = 90)
    auc = roc_auc_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    print("AUC: {:.3f}".format(auc))
    return pd.DataFrame(report) ,pd.DataFrame(cm)



def scorer_if(estimator, X,percentile_flag):
    return np.percentile(estimator.score_samples(X),percentile_flag)


@timer_decorator
def tune_isolation_forest(X_train, y_train, param_grid=None, scoring=scorer_if ,n_iter=10, cv=5, random_state=42):
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
    if scoring is None:
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
