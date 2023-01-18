import pandas as pd
from src.preprocessing.helpers import timer_decorator


def read_data(path):
    data = pd.read_pickle(path)
    return data

@timer_decorator
def read_all_trx(files_path,axis = 0):
    total_data = pd.concat( [ read_data(file) for file in files_path ] ,axis=axis)
    return total_data


