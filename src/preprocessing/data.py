import pandas as pd
from src.preprocessing.helpers import timer_decorator


def read_data(path):
    data = pd.read_pickle(path)
    return data

@timer_decorator
def read_all_trx(files_path,axis = 0):
    total_data = pd.concat( [ read_data(file) for file in files_path ] ,axis=axis)
    return total_data


def train_test_split_transactions(X,features,train_start= "2018-04-01",train_end = "2018-07-01",test_start = "2018-07-01",test_end = "2018-09-01",target = "TX_FRAUD"):
    """
    Parameters:
    X (pd.DataFrame) : The input dataset containing the transactions data.
    features (List[str]) : List of features columns names
    target (str) : Target column name
    train_start (str) : Start time for the training set in the format 'YYYY-MM-DD'
    train_end (str) : End time for the training set in the format 'YYYY-MM-DD'
    test_start (str) : Start time for the testing set in the format 'YYYY-MM-DD'
    test_end (str) : End time for the testing set in the format 'YYYY-MM-DD'

    Returns:
    Tuple : Tuple containing the splitted data (X_train,X_test,y_train,y_test)
    """
    X_train = X.loc[X["TX_DATETIME"].between(train_start,train_end)][features]
    y_train = X.loc[X["TX_DATETIME"].between(train_start,train_end)][target]

    X_test = X.loc[X["TX_DATETIME"].between(test_start,test_end)][features]
    y_test = X.loc[X["TX_DATETIME"].between(test_start,test_end)][target]

    return X_train,X_test,y_train,y_test

