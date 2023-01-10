import pandas as pd
from src.preprocessing.helpers import timer_decorator


def read_data(path):
    data = pd.read_pickle(path)
    return data

@timer_decorator
def read_all_trx(files_path,axis = 0):
    total_data = pd.concat( [ read_data(file) for file in files_path ] ,axis=axis)
    return total_data

def is_night(tx_datetime):

    # Get the hour of the transaction
    tx_hour = tx_datetime.hour
    # Binary value: 1 if hour less than 6, and 0 otherwise
    is_night = tx_hour <= 6

    return int(is_night)


def is_weekend(tx_datetime):
    # Transform date into weekday (0 is Monday, 6 is Sunday)
    weekday = tx_datetime.weekday()
    # Binary value: 0 if weekday, 1 if weekend
    is_weekend = weekday >= 5

    return int(is_weekend)

