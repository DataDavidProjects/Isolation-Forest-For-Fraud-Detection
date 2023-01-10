import pandas as pd
import sys
sys.path.append("/Isolation-Forest-For-Fraud-Detection/src/")
from src.preprocessing.data import read_data,read_all_trx
from src.preprocessing.features import get_customer_spending_behaviour_features,get_count_risk_rolling_window

calendar = pd.date_range('2018-04-01', '2018-09-30',  inclusive="both").strftime('%Y-%m-%d')
root = "https://github.com/Fraud-Detection-Handbook/simulated-data-raw/blob/main/data/"
path_data = [f"{root}{date}.pkl?raw=true" for date in calendar]


transactions_df = read_all_trx(path_data).sort_values('TX_DATETIME').reset_index(drop=True)


X = create_feature_matrix(transactions_df)
