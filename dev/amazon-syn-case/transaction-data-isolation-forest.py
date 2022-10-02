import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin

path = "C:/Users/david/Desktop/Projects/Isolation-Forest-For-Fraud-Detection/data/transaction_data_100K_full.csv"
data = pd.read_csv(path,parse_dates=["EVENT_TIMESTAMP"]).sort_values(by="EVENT_TIMESTAMP").reset_index()

#https://docs.aws.amazon.com/frauddetector/latest/ug/create-event-dataset.html#prepare-event-dataset
features= ['EVENT_TIMESTAMP',
           'card_bin', 'customer_name',
           'billing_street', 'billing_city', 'billing_state', 'billing_zip',
           'billing_latitude', 'billing_longitude', 'customer_job', 'ip_address',
           'customer_email', 'phone', 'user_agent', 'product_category',
           'order_price', 'payment_currency', 'merchant']
X = data.loc[:,features]
y = data.loc[:,['EVENT_LABEL']]

class DateEncoder(BaseEstimator, TransformerMixin):
    '''
     Parameters-Fixed
    ----------
        time frame dates  ["year", "month", "day", "hour", "minute", "second"]
    Returns
    -------
    for each date column:
            year , month , day ,hour , minute, seconds in seperate columns

    '''

    def __init__(self, by=1, columns=None):
        self.by = by
        self.columns = columns
        assert type(X) == "<class 'pandas.core.frame.DataFrame'>"

    def fit(self, X, y=None):
        columns = X.columns if self.columns is None else self.columns
        return self

    def transform(self,X, y=None):
        columns = X.columns if self.columns is None else self.columns
        output = X.copy()
        # Converting in datetime format
        time_frames_str = ["year", "month", "day", "hour", "minute", "second"]
        for col in columns:
            # cast as time
            output[col] = pd.to_datetime(output[col])
            for time in time_frames_str:
                try:
                    # create the new column with time frame information
                    output[col+"_"+time] = getattr(output[col].dt, time)
                except:
                    print(f"Problems in {col,time}")
                    pass
            # drop original columns
            output = output.drop(col,axis = 1)
        return output