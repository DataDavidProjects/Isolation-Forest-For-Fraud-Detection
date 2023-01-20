import time
import pandas as pd
import scipy as sp
import numpy as np
from scipy.stats import chi2

def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f'Execution time: {execution_time} seconds')
        return result
    return wrapper





