from src.preprocessing.helpers import timer_decorator


def get_customer_spending_behaviour_features(customer_transactions, windows_size_in_days=[1, 7, 30]):
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


def get_count_risk_rolling_window(terminal_transactions, delay_period=7, windows_size_in_days=[1, 7, 30],feature="TERMINAL_ID"):
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


def create_features(transactions_df,key = 'CUSTOMER_ID',pipeline="customer",windows_size_in_days=[1, 7, 30],delay_period=7):
    if pipeline=="customer":

        transactions_df = transactions_df.groupby(key).apply(lambda x: get_customer_spending_behaviour_features(x, windows_size_in_days=windows_size_in_days)).reset_index(drop=True)

    elif pipeline=="terminal":
        transactions_df = transactions_df.groupby(key).apply(lambda x: get_count_risk_rolling_window(x, delay_period=delay_period, windows_size_in_days=windows_size_in_days,feature="TERMINAL_ID"))
    else:
        raise  AttributeError(f"Pipeline '{pipeline}' does not exist")

    return transactions_df.reset_index(drop=True)

@timer_decorator
def create_feature_matrix(transactions_df,windows_size_in_days=[1, 7, 30],delay_period=7):
    customer_features = create_features(create_features(transactions_df,key = 'CUSTOMER_ID',pipeline="customer",windows_size_in_days=windows_size_in_days))
    terminal_features = create_features(create_features(transactions_df,key = 'TERMINAL_ID',pipeline="terminal",windows_size_in_days=windows_size_in_days,delay_period=delay_period))
    X = customer_features.merge(terminal_features, on = "TRANSACTION_ID")
    return X
