import pandas as pd
from src.preprocessing.helpers import timer_decorator

def is_night(tx_datetime):
    # Get the hour of the transaction
    tx_hour = tx_datetime.hour
    # Binary value: opposite of day ( day is hour between 6 and 18)
    is_night = not(tx_hour >= 6 and tx_hour <= 18)
    return int(is_night)


def is_weekend(tx_datetime):
    # Transform date into weekday (0 is Monday, 6 is Sunday)
    weekday = tx_datetime.weekday()
    # Binary value: 0 if weekday, 1 if weekend
    is_weekend = weekday >= 5
    return int(is_weekend)


def get_customer_spending_behaviour_features(customer_transactions, windows_size_in_days=[1, 7, 30]):
    """
    Compute features for a customer's spending behaviour based on transaction data and a list of rolling window sizes.
    The features include the number of transactions and the average transaction amount in the given rolling window.

    Parameters:
        customer_transactions (pd.DataFrame): DataFrame containing transaction data for a single customer
        windows_size_in_days (list[int]): list of integers representing rolling window sizes in days

    Returns:
        pd.DataFrame: DataFrame containing the original data and added features for the customer's spending behaviour
    """

    # Let us first order transactions chronologically
    customer_transactions = customer_transactions.sort_values('TX_DATETIME')

    # The transaction date and time is set as the index, which will allow the use of the rolling function
    customer_transactions.index = customer_transactions.TX_DATETIME

    # For each window size
    for window_size in windows_size_in_days:
        # Compute statistics of the transaction amounts for the given window size
        SUM_AMOUNT_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(str(window_size) + 'd').sum()
        NB_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(str(window_size) + 'd').count()
        MAX_AMOUNT_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(str(window_size) + 'd').max()
        STD_AMOUNT_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(str(window_size) + 'd').std()
        QNT70_AMOUNT_TX_WINDOW = customer_transactions.set_index("TX_DATETIME")['TX_AMOUNT'].rolling(str(3) + 'd').quantile(0.7)
        QNT90_AMOUNT_TX_WINDOW = customer_transactions.set_index("TX_DATETIME")['TX_AMOUNT'].rolling(str(3) + 'd').quantile(0.9)


        AVG_AMOUNT_TX_WINDOW = SUM_AMOUNT_TX_WINDOW / NB_TX_WINDOW
        VC_AMOUNT_TX_WINDOW = STD_AMOUNT_TX_WINDOW/AVG_AMOUNT_TX_WINDOW * 100

        # Save feature values
        customer_transactions['CUSTOMER_ID_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)
        customer_transactions['CUSTOMER_ID_AVG_AMOUNT_' + str(window_size) + 'DAY_WINDOW'] = list(AVG_AMOUNT_TX_WINDOW)
        customer_transactions['CUSTOMER_ID_MAX_AMOUNT_' + str(window_size) + 'DAY_WINDOW'] = list(MAX_AMOUNT_TX_WINDOW)
        customer_transactions['CUSTOMER_ID_VC_AMOUNT_' + str(window_size) + 'DAY_WINDOW'] = list(VC_AMOUNT_TX_WINDOW)
        customer_transactions['CUSTOMER_ID_QT70_AMOUNT_' + str(window_size) + 'DAY_WINDOW'] = list(QNT70_AMOUNT_TX_WINDOW)
        customer_transactions['CUSTOMER_ID_QT90_AMOUNT_' + str(window_size) + 'DAY_WINDOW'] = list(QNT90_AMOUNT_TX_WINDOW)

    # Flags amount compared to rolling tx distributions
    for window_size in windows_size_in_days:
        customer_transactions[f"TX_FLAG_AMOUNT_IS_MAX_" + str(window_size) + 'DAY_WINDOW'] = (customer_transactions["TX_AMOUNT"] >=  customer_transactions['CUSTOMER_ID_MAX_AMOUNT_' + str(window_size) + 'DAY_WINDOW']).astype(int)
        for q in [0.7,0.9]:
            customer_transactions[f"TX_FLAG_QT{str(int(q*100)).upper()}_"+ str(window_size)+ 'DAY_WINDOW'] = (customer_transactions["TX_AMOUNT"] >= customer_transactions[f'CUSTOMER_ID_QT{str(int(q*100)).upper()}_AMOUNT_' + str(window_size) + 'DAY_WINDOW']).astype(int)

    # Reindex according to transaction IDs
    customer_transactions = customer_transactions.set_index("TRANSACTION_ID")

    # And return the dataframe with the new features
    return customer_transactions.reset_index(names="TRANSACTION_ID").fillna(-1)

def get_count_risk_rolling_window(terminal_transactions, delay_period= 7, windows_size_in_days= [1, 7, 30], feature= "TERMINAL_ID"):
    """
    Computes rolling window count and risk for fraud transactions for a given terminal.

    Parameters:
        terminal_transactions : pd.DataFrame : DataFrame containing transactions data for a specific terminal
        delay_period : int : number of days to delay calculation of fraud transactions
        windows_size_in_days : List[int] : list of integers that represent the rolling window size in days, used to create features dataframes
        feature : str : the column name to use in feature column names

    Returns:
        pd.DataFrame : DataFrame containing generated features (Number of transactions and risk of fraud)

    """
    # Sort the transactions data by the 'TX_DATETIME' column
    terminal_transactions = terminal_transactions.sort_values('TX_DATETIME')

    # Assign the 'TX_DATETIME' column as the index of the dataframe
    terminal_transactions.index = terminal_transactions.TX_DATETIME

    # Create a new column 'NB_FRAUD_DELAY' which is the rolling sum of the 'TX_FRAUD' column for the 'delay_period' days
    NB_FRAUD_DELAY = terminal_transactions['TX_FRAUD'].rolling(str(delay_period) + 'd').sum()
    NB_TX_DELAY = terminal_transactions['TX_FRAUD'].rolling(str(delay_period) + 'd').count()

    # Iterate over the 'windows_size_in_days' list
    for window_size in windows_size_in_days:
        # Create a new column 'NB_FRAUD_DELAY_WINDOW'
        # which is the rolling sum of the 'TX_FRAUD' column for the 'delay_period + window_size' days
        NB_FRAUD_DELAY_WINDOW = terminal_transactions['TX_FRAUD'].rolling(str(delay_period + window_size) + 'd').sum()
        NB_TX_DELAY_WINDOW = terminal_transactions['TX_FRAUD'].rolling(str(delay_period + window_size) + 'd').count()

        # Create a new column 'NB_FRAUD_WINDOW' which is the difference between 'NB_FRAUD_DELAY_WINDOW' and 'NB_FRAUD_DELAY'
        NB_FRAUD_WINDOW = NB_FRAUD_DELAY_WINDOW - NB_FRAUD_DELAY
        NB_TX_WINDOW = NB_TX_DELAY_WINDOW - NB_TX_DELAY

        # Create a new column 'RISK_WINDOW' which is the division of 'NB_FRAUD_WINDOW' by 'NB_TX_WINDOW'
        RISK_WINDOW = NB_FRAUD_WINDOW / NB_TX_WINDOW

        # Create new columns in the dataframe, with the desired format, using the calculated values
        terminal_transactions[feature + '_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)
        terminal_transactions[feature + '_RISK_' + str(window_size) + 'DAY_WINDOW'] = list(RISK_WINDOW)

    # Assign the 'TRANSACTION_ID' column as the index of the dataframe
    terminal_transactions = terminal_transactions.set_index("TRANSACTION_ID")


    # Replace NA values with 0 (all undefined risk scores where NB_TX_WINDOW is 0)
    terminal_transactions.fillna(0, inplace=True)

    return terminal_transactions.reset_index(names="TRANSACTION_ID")

def get_time_features(transactions_df, time_col= "TX_DATETIME", time_frames =["month", "day", "hour", "minute"], index="TRANSACTION_ID"):
    """
       Extract time-based features from a DataFrame containing transaction data.

       Parameters:
           transactions_df : pd.DataFrame : DataFrame containing transactions data
           time_col : str : Column name of the datetime column
           time_frames : list : list of strings representing the time components to extract ["day","hour", "minute"]
           index : str : Column name of the index column

       Returns:
           pd.DataFrame : DataFrame containing extracted time-based features.
    """
    # Cast the columns as a datetime object
    transactions_df[time_col] = pd.to_datetime(transactions_df[time_col]) # make sure the column is datetime
    # Sort by time
    transactions_df= transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
    # Iterate over the column to extract month day hour minute seconds
    time_feats = pd.concat([transactions_df[time_col].dt.__getattribute__(time) for time in time_frames], axis=1)
    # Rename the columns using upper case
    time_feats.columns = ["TX_"+time.upper() for time in time_frames]
    # Get dummy features based on time of trx
    time_feats['TX_DURING_WEEKEND'] = transactions_df.TX_DATETIME.apply(is_weekend)
    time_feats['TX_DURING_NIGHT'] = transactions_df.TX_DATETIME.apply(is_night)
    # Add back TRANSACTION_ID
    time_feats["TRANSACTION_ID"] = transactions_df["TRANSACTION_ID"]
    return time_feats

@timer_decorator
def create_features(transactions_df, key='CUSTOMER_ID', pipeline= "customer", windows_size_in_days= [1, 7, 30],delay_period= 7):
    """
    Create features dataframe based on the provided pipeline.

    Parameters:
        transactions_df : pd.DataFrame : DataFrame containing transactions data
        key : str : Column used to group the dataframe by
        pipeline : str : String representing the pipeline to use, should be one of ["customer", "terminal", "time"]
        windows_size_in_days : List[int] : list of integers that represent the rolling window size in days, used to create features dataframes
        delay_period : int : integer that represent the number of days to delay feature calculation

    Returns:
        pd.DataFrame : DataFrame containing generated features

    """
    # Validate pipeline name
    if pipeline not in ["customer", "terminal", "time"]:
        raise AttributeError(f"Pipeline '{pipeline}' does not exist")
    # Apply different pipelines based on the input pipeline name
    if pipeline == "customer":
        return transactions_df.groupby(key).apply(lambda x: get_customer_spending_behaviour_features(x,windows_size_in_days=windows_size_in_days)).reset_index(drop=True)

    if pipeline == "terminal":
        return transactions_df.groupby(key).apply(lambda x: get_count_risk_rolling_window(x, delay_period=delay_period,windows_size_in_days=windows_size_in_days,feature="TERMINAL_ID")).reset_index(drop=True)

    if pipeline == "time":
        return get_time_features(transactions_df, key)


@timer_decorator
def create_feature_matrix(transactions_df, windows_size_in_days= [1, 7, 30], delay_period= 7):
    """
    Create features matrix by concatenating features dataframes generated by different pipelines.

    Parameters:
        transactions_df : pd.DataFrame : DataFrame containing transactions data
        windows_size_in_days : List[int] : list of integers that represent the rolling window size in days, used to create features dataframes
        delay_period : int : integer that represent the number of days to delay feature calculation

    Returns :
        pd.DataFrame : DataFrame containing generated features

    """
    # Sort by time
    transactions_df = transactions_df.sort_values('TX_DATETIME').reset_index(drop=True)
    feature_df_list = []
    # Iterate over different pipelines and keys to generate features dataframe
    for pipeline, key in [("customer","CUSTOMER_ID"),("terminal","TERMINAL_ID"),("time","TX_DATETIME")]:
        features_df = create_features(transactions_df, key=key, pipeline=pipeline, windows_size_in_days=windows_size_in_days, delay_period=delay_period)
        feature_df_list.append(features_df.set_index("TRANSACTION_ID"))
    # Concatenate features dataframes and prevent duplicates
    X = pd.concat(feature_df_list, axis=1)
    return X.loc[:, ~X.columns.duplicated()]


