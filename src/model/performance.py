from sklearn.metrics import confusion_matrix,make_scorer,roc_auc_score,classification_report
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    y_pred = [1 if s > threshold else 0 for s in scores]
    return y_pred

def evaluate_model(model, X_test, y_test,plot = True, threshold = 99 ):
    """
        Evaluates the performance of the model using AUC and classification report

        Parameters:
        model (object): The trained model that will be used to make predictions
        X_test (pd.DataFrame): The test set
        y_test (pd.Series): The true labels of the test set
        plot (bool): Plot Confusion Matrix

        Returns:
        pd.DataFrame: A dataframe containing the classification report, benchmark and CM
    """
    # Note that model score are flipped for scikit learn evaluation greater better
    scores = -model.score_samples(X_test)
    alert = np.percentile(scores,threshold)
    y_pred = (scores > alert).astype(int)
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    # Make predictions using model,random and dummy
    report = classification_report(y_test, y_pred, output_dict=True)
    random_scores = pd.Series(np.random.uniform(size=len(y_test)))
    dummy_not = [0 for _ in range(0, len(y_test))]
    # Summarize in benchmark
    benchmark = {
       "model":  roc_auc_score(y_test, scores),
       "dummy":  roc_auc_score(y_test, dummy_not),
       "random": roc_auc_score(y_test, random_scores)
    }
    auc = roc_auc_score(y_test, scores)
    print("AUC: {:.3f}".format(auc))
    # Plot CM
    if plot:
        sns.heatmap(cm, annot=True, fmt="d", cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.show()
    return pd.DataFrame(report), pd.DataFrame(cm), pd.Series(benchmark)



def scorer_if(estimator, X,percentile_flag):
    """
    :param estimator: isolation forest
    :param X: feature matrix
    :param percentile_flag: percentile of anomaly score chosen as alert
    :return: returns the value of the percentile
    """
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
