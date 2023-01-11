import pandas as pd
from src.preprocessing.helpers import timer_decorator


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
        # Compute the sum of the transaction amounts and the number of transactions for the given window size
        SUM_AMOUNT_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(str(window_size) + 'd').sum()
        NB_TX_WINDOW = customer_transactions['TX_AMOUNT'].rolling(str(window_size) + 'd').count()

        # Compute the average transaction amount for the given window size
        # NB_TX_WINDOW is always >0 since current transaction is always included
        AVG_AMOUNT_TX_WINDOW = SUM_AMOUNT_TX_WINDOW / NB_TX_WINDOW

        # Save feature values
        customer_transactions['CUSTOMER_ID_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)
        customer_transactions['CUSTOMER_ID_AVG_AMOUNT_' + str(window_size) + 'DAY_WINDOW'] = list(AVG_AMOUNT_TX_WINDOW)

    # Reindex according to transaction IDs
    customer_transactions.index = customer_transactions.TRANSACTION_ID

    # And return the dataframe with the new features
    return customer_transactions

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
    terminal_transactions = terminal_transactions.sort_values('TX_DATETIME')

    terminal_transactions.index = terminal_transactions.TX_DATETIME

    NB_FRAUD_DELAY = terminal_transactions['TX_FRAUD'].rolling(str(delay_period) + 'd').sum()
    NB_TX_DELAY = terminal_transactions['TX_FRAUD'].rolling(str(delay_period) + 'd').count()

    for window_size in windows_size_in_days:
        NB_FRAUD_DELAY_WINDOW = terminal_transactions['TX_FRAUD'].rolling(str(delay_period + window_size) + 'd').sum()
        NB_TX_DELAY_WINDOW = terminal_transactions['TX_FRAUD'].rolling(str(delay_period + window_size) + 'd').count()

        NB_FRAUD_WINDOW = NB_FRAUD_DELAY_WINDOW - NB_FRAUD_DELAY
        NB_TX_WINDOW = NB_TX_DELAY_WINDOW - NB_TX_DELAY

        RISK_WINDOW = NB_FRAUD_WINDOW / NB_TX_WINDOW

        terminal_transactions[feature + '_NB_TX_' + str(window_size) + 'DAY_WINDOW'] = list(NB_TX_WINDOW)
        terminal_transactions[feature + '_RISK_' + str(window_size) + 'DAY_WINDOW'] = list(RISK_WINDOW)

    terminal_transactions.index = terminal_transactions.TRANSACTION_ID

    # Replace NA values with 0 (all undefined risk scores where NB_TX_WINDOW is 0)
    terminal_transactions.fillna(0, inplace=True)

    return terminal_transactions

def get_time_features(transactions_df,time_col= "TX_DATETIME",time_frames = ["day","hour", "minute"],index="TRANSACTION_ID"):
    transactions_df = transactions_df[[time_col,index]].copy()
    transactions_df[time_col] = pd.to_datetime(transactions_df[time_col]) # make sure the column is datetime
    time_feats = pd.concat([transactions_df[time_col].dt.__getattribute__(time) for time in time_frames], axis=1)
    time_feats.columns = ["TX_"+time.upper() for time in time_frames]
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
    feature_df_list = []
    # Iterate over different pipelines and keys to generate features dataframe
    for pipeline, key in [("customer","CUSTOMER_ID"),("terminal","TERMINAL_ID"),("time","TX_DATETIME")]:
        features_df = create_features(transactions_df, key=key, pipeline=pipeline, windows_size_in_days=windows_size_in_days, delay_period=delay_period)
        feature_df_list.append(features_df)
    # Concatenate features dataframes and prevent duplicates
    X = pd.concat(feature_df_list, axis=1)
    return X.loc[:, ~X.columns.duplicated()]
