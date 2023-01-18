import pandas as pd
import numpy as np
import time
import datetime
from sklearn import metrics
import sklearn


def get_train_test_set(transactions_df,
                       start_date_training,
                       delta_train=7, delta_delay=7, delta_test=7,
                       sampling_ratio=1.0,
                       random_state=0):
    # Get the training set data
    train_df = transactions_df[(transactions_df.TX_DATETIME >= start_date_training) &
                               (transactions_df.TX_DATETIME < start_date_training + datetime.timedelta(
                                   days=delta_train))]

    # Get the test set data
    test_df = []

    # Note: Cards known to be compromised after the delay period are removed from the test set
    # That is, for each test day, all frauds known at (test_day-delay_period) are removed

    # First, get known defrauded customers from the training set
    known_defrauded_customers = set(train_df[train_df.TX_FRAUD == 1].CUSTOMER_ID)

    # Get the relative starting day of training set (easier than TX_DATETIME to collect test data)
    start_tx_time_days_training = train_df.TX_TIME_DAYS.min()

    # Then, for each day of the test set
    for day in range(delta_test):
        # Get test data for that day
        test_df_day = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                      delta_train + delta_delay +
                                      day]

        # Compromised cards from that test day, minus the delay period, are added to the pool of known defrauded customers
        test_df_day_delay_period = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                                   delta_train +
                                                   day - 1]

        new_defrauded_customers = set(test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD == 1].CUSTOMER_ID)
        known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)

        test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]

        test_df.append(test_df_day)

    test_df = pd.concat(test_df)

    # If subsample
    if sampling_ratio < 1:
        train_df_frauds = train_df[train_df.TX_FRAUD == 1].sample(frac=sampling_ratio, random_state=random_state)
        train_df_genuine = train_df[train_df.TX_FRAUD == 0].sample(frac=sampling_ratio, random_state=random_state)
        train_df = pd.concat([train_df_frauds, train_df_genuine])

    # Sort data sets by ascending order of transaction ID
    train_df = train_df.sort_values('TRANSACTION_ID')
    test_df = test_df.sort_values('TRANSACTION_ID')

    return (train_df, test_df)



def get_train_delay_test_set(transactions_df,
                             start_date_training,
                             delta_train=7, delta_delay=7, delta_test=7,
                             sampling_ratio=1.0,
                             random_state=0):
    # Get the training set data
    train_df = transactions_df[(transactions_df.TX_DATETIME >= start_date_training) &
                               (transactions_df.TX_DATETIME < start_date_training + datetime.timedelta(
                                   days=delta_train))]

    # Get the delay set data
    delay_df = transactions_df[
        (transactions_df.TX_DATETIME >= start_date_training + datetime.timedelta(days=delta_train)) &
        (transactions_df.TX_DATETIME < start_date_training + datetime.timedelta(days=delta_train) +
         +datetime.timedelta(days=delta_delay))]

    # Get the test set data
    test_df = []

    # Note: Cards known to be compromised after the delay period are removed from the test set
    # That is, for each test day, all frauds known at (test_day-delay_period) are removed

    # First, get known defrauded customers from the training set
    known_defrauded_customers = set(train_df[train_df.TX_FRAUD == 1].CUSTOMER_ID)

    # Get the relative starting day of training set (easier than TX_DATETIME to collect test data)
    start_tx_time_days_training = train_df.TX_TIME_DAYS.min()

    # Then, for each day of the test set
    for day in range(delta_test):
        # Get test data for that day
        test_df_day = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                      delta_train + delta_delay +
                                      day]

        # Compromised cards from that test day, minus the delay period, are added to the pool of known defrauded customers
        test_df_day_delay_period = transactions_df[transactions_df.TX_TIME_DAYS == start_tx_time_days_training +
                                                   delta_train +
                                                   day - 1]

        new_defrauded_customers = set(test_df_day_delay_period[test_df_day_delay_period.TX_FRAUD == 1].CUSTOMER_ID)
        known_defrauded_customers = known_defrauded_customers.union(new_defrauded_customers)

        test_df_day = test_df_day[~test_df_day.CUSTOMER_ID.isin(known_defrauded_customers)]

        test_df.append(test_df_day)

    test_df = pd.concat(test_df)

    # If subsample
    if sampling_ratio < 1:
        train_df_frauds = train_df[train_df.TX_FRAUD == 1].sample(frac=sampling_ratio, random_state=random_state)
        train_df_genuine = train_df[train_df.TX_FRAUD == 0].sample(frac=sampling_ratio, random_state=random_state)
        train_df = pd.concat([train_df_frauds, train_df_genuine])

    # Sort data sets by ascending order of transaction ID
    train_df = train_df.sort_values('TRANSACTION_ID')
    test_df = test_df.sort_values('TRANSACTION_ID')

    return (train_df, delay_df, test_df)


def prequentialSplit(transactions_df,
                     start_date_training,
                     n_folds=4,
                     delta_train=7,
                     delta_delay=7,
                     delta_assessment=7):
    prequential_split_indices = []

    # For each fold
    for fold in range(n_folds):
        # Shift back start date for training by the fold index times the assessment period (delta_assessment)
        # (See Fig. 5)
        start_date_training_fold = start_date_training - datetime.timedelta(days=fold * delta_assessment)

        # Get the training and test (assessment) sets
        (train_df, test_df) = get_train_test_set(transactions_df,
                                                 start_date_training=start_date_training_fold,
                                                 delta_train=delta_train, delta_delay=delta_delay,
                                                 delta_test=delta_assessment)

        # Get the indices from the two sets, and add them to the list of prequential splits
        indices_train = list(train_df.index)
        indices_test = list(test_df.index)

        prequential_split_indices.append((indices_train, indices_test))

    return prequential_split_indices


def card_precision_top_k_day(df_day, top_k):
    # This takes the max of the predictions AND the max of label TX_FRAUD for each CUSTOMER_ID,
    # and sorts by decreasing order of fraudulent prediction
    df_day = df_day.groupby('CUSTOMER_ID').max().sort_values(by="predictions", ascending=False).reset_index(drop=False)

    # Get the top k most suspicious cards
    df_day_top_k = df_day.head(top_k)
    list_detected_compromised_cards = list(df_day_top_k[df_day_top_k.TX_FRAUD == 1].CUSTOMER_ID)

    # Compute precision top k
    card_precision_top_k = len(list_detected_compromised_cards) / top_k

    return list_detected_compromised_cards, card_precision_top_k


def card_precision_top_k(predictions_df, top_k, remove_detected_compromised_cards=True):
    # Sort days by increasing order
    list_days = list(predictions_df['TX_TIME_DAYS'].unique())
    list_days.sort()

    # At first, the list of detected compromised cards is empty
    list_detected_compromised_cards = []

    card_precision_top_k_per_day_list = []
    nb_compromised_cards_per_day = []

    # For each day, compute precision top k
    for day in list_days:

        df_day = predictions_df[predictions_df['TX_TIME_DAYS'] == day]
        df_day = df_day[['predictions', 'CUSTOMER_ID', 'TX_FRAUD']]

        # Let us remove detected compromised cards from the set of daily transactions
        df_day = df_day[df_day.CUSTOMER_ID.isin(list_detected_compromised_cards) == False]

        nb_compromised_cards_per_day.append(len(df_day[df_day.TX_FRAUD == 1].CUSTOMER_ID.unique()))

        detected_compromised_cards, card_precision_top_k = card_precision_top_k_day(df_day, top_k)

        card_precision_top_k_per_day_list.append(card_precision_top_k)

        # Let us update the list of detected compromised cards
        if remove_detected_compromised_cards:
            list_detected_compromised_cards.extend(detected_compromised_cards)

    # Compute the mean
    mean_card_precision_top_k = np.array(card_precision_top_k_per_day_list).mean()

    # Returns precision top k per day as a list, and resulting mean
    return nb_compromised_cards_per_day, card_precision_top_k_per_day_list, mean_card_precision_top_k



def card_precision_top_k_custom(y_true, y_pred, top_k, transactions_df):
    # Let us create a predictions_df DataFrame, that contains all transactions matching the indices of the current fold
    # (indices of the y_true vector)
    predictions_df = transactions_df.iloc[y_true.index.values].copy()
    predictions_df['predictions'] = y_pred

    # Compute the CP@k using the function implemented in Chapter 4, Section 4.2
    nb_compromised_cards_per_day, card_precision_top_k_per_day_list, mean_card_precision_top_k = card_precision_top_k(
        predictions_df, top_k)

    # Return the mean_card_precision_top_k
    return mean_card_precision_top_k


def performance_assessment(predictions_df, output_feature='TX_FRAUD',
                           prediction_feature='predictions', top_k_list=[100],
                           rounded=True):
    AUC_ROC = metrics.roc_auc_score(predictions_df[output_feature], predictions_df[prediction_feature])
    AP = metrics.average_precision_score(predictions_df[output_feature], predictions_df[prediction_feature])

    performances = pd.DataFrame([[AUC_ROC, AP]],
                                columns=['AUC ROC', 'Average precision'])

    for top_k in top_k_list:
        _, _, mean_card_precision_top_k = card_precision_top_k(predictions_df, top_k)
        performances['Card Precision@' + str(top_k)] = mean_card_precision_top_k

    if rounded:
        performances = performances.round(3)

    return performances


def performance_assessment_model_collection(fitted_models_and_predictions_dictionary,
                                            transactions_df,
                                            type_set='test',
                                            top_k_list=[100]):
    performances = pd.DataFrame()

    for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
        predictions_df = transactions_df

        predictions_df['predictions'] = model_and_predictions['predictions_' + type_set]

        performances_model = performance_assessment(predictions_df, output_feature='TX_FRAUD',
                                                    prediction_feature='predictions', top_k_list=top_k_list)
        performances_model.index = [classifier_name]

        performances = performances.append(performances_model)

    return performances



def execution_times_model_collection(fitted_models_and_predictions_dictionary):
    execution_times = pd.DataFrame()

    for classifier_name, model_and_predictions in fitted_models_and_predictions_dictionary.items():
        execution_times_model = pd.DataFrame()
        execution_times_model['Training execution time'] = [model_and_predictions['training_execution_time']]
        execution_times_model['Prediction execution time'] = [model_and_predictions['prediction_execution_time']]
        execution_times_model.index = [classifier_name]

        execution_times = execution_times.append(execution_times_model)

    return execution_times


# Getting classes from a vector of fraud probabilities and a threshold
def get_class_from_fraud_probability(fraud_probabilities, threshold=0.5):
    predicted_classes = [0 if fraud_probability < threshold else 1
                         for fraud_probability in fraud_probabilities]

    return predicted_classes


def threshold_based_metrics(fraud_probabilities, true_label, thresholds_list):
    results = []

    for threshold in thresholds_list:

        predicted_classes = get_class_from_fraud_probability(fraud_probabilities, threshold=threshold)

        (TN, FP, FN, TP) = metrics.confusion_matrix(true_label, predicted_classes).ravel()

        MME = (FP + FN) / (TN + FP + FN + TP)

        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)

        FPR = FP / (TN + FP)
        FNR = FN / (TP + FN)

        BER = 1 / 2 * (FPR + FNR)

        Gmean = np.sqrt(TPR * TNR)

        precision = 1  # 1 if TP+FP=0
        FDR = 1  # 1 if TP+FP=0

        if TP + FP > 0:
            precision = TP / (TP + FP)
            FDR = FP / (TP + FP)

        NPV = 1  # 1 if TN+FN=0
        FOR = 1  # 1 if TN+FN=0

        if TN + FN > 0:
            NPV = TN / (TN + FN)
            FOR = FN / (TN + FN)

        F1_score = 2 * (precision * TPR) / (precision + TPR)

        results.append([threshold, MME, TPR, TNR, FPR, FNR, BER, Gmean, precision, NPV, FDR, FOR, F1_score])

    results_df = pd.DataFrame(results,
                              columns=['Threshold', 'MME', 'TPR', 'TNR', 'FPR', 'FNR', 'BER', 'G-mean', 'Precision',
                                       'NPV', 'FDR', 'FOR', 'F1 Score'])

    return results_df


def get_summary_performances(performances_df, parameter_column_name="Parameters summary"):
    metrics = ['AUC ROC', 'Average precision', 'Card Precision@100']
    performances_results = pd.DataFrame(columns=metrics)

    performances_df.reset_index(drop=True, inplace=True)

    best_estimated_parameters = []
    validation_performance = []
    test_performance = []

    for metric in metrics:
        index_best_validation_performance = performances_df.index[
            np.argmax(performances_df[metric + ' Validation'].values)]

        best_estimated_parameters.append(performances_df[parameter_column_name].iloc[index_best_validation_performance])

        validation_performance.append(
            str(round(performances_df[metric + ' Validation'].iloc[index_best_validation_performance], 3)) +
            '+/-' +
            str(round(performances_df[metric + ' Validation' + ' Std'].iloc[index_best_validation_performance], 2))
        )

        test_performance.append(
            str(round(performances_df[metric + ' Test'].iloc[index_best_validation_performance], 3)) +
            '+/-' +
            str(round(performances_df[metric + ' Test' + ' Std'].iloc[index_best_validation_performance], 2))
        )

    performances_results.loc["Best estimated parameters"] = best_estimated_parameters
    performances_results.loc["Validation performance"] = validation_performance
    performances_results.loc["Test performance"] = test_performance

    optimal_test_performance = []
    optimal_parameters = []

    for metric in ['AUC ROC Test', 'Average precision Test', 'Card Precision@100 Test']:
        index_optimal_test_performance = performances_df.index[np.argmax(performances_df[metric].values)]

        optimal_parameters.append(performances_df[parameter_column_name].iloc[index_optimal_test_performance])

        optimal_test_performance.append(
            str(round(performances_df[metric].iloc[index_optimal_test_performance], 3)) +
            '+/-' +
            str(round(performances_df[metric + ' Std'].iloc[index_optimal_test_performance], 2))
        )

    performances_results.loc["Optimal parameter(s)"] = optimal_parameters
    performances_results.loc["Optimal test performance"] = optimal_test_performance

    return performances_results