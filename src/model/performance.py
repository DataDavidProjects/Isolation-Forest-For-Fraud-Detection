from sklearn.metrics import confusion_matrix,roc_auc_score,classification_report
import numpy as np
import pandas as pd

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
