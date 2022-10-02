import pandas as pd
import numpy as np
import time

data = pd.read_csv("data/card_transaction.v1.csv")

def clean_data(df):
    dummies_error = pd.get_dummies(df["Errors?"].replace([np.nan],["Regular"]).apply(lambda x:x.split(",")).explode().replace("",np.nan).dropna(axis = 0))
    df = df.join(dummies_error).drop({"Errors?"},axis=1)
    df["Is Fraud?"] = [0 if i=="No" else 1 for i in df["Is Fraud?"].values]
    df["Merchant State"] = df["Merchant State"].fillna("ONLINE")
    df["Zip"] = df["Zip"].fillna("ONLINE")
    df["Amount"] = df["Amount"].replace({'\$':''}, regex = True).astype(float)
    df["Hour"] = pd.to_datetime(df["Time"]).dt.hour
    df["Minute"] = pd.to_datetime(df["Time"]).dt.minute
    df["Transaction-Time"] = pd.to_datetime(df[["Year", "Month", "Day", "Time"]].astype(str).agg('-'.join, axis=1),format='%Y-%m-%d-%H:%M')
    df = df.drop({"Time"},axis=1)
    df = df[['Transaction-Time', 'Year', 'Month', 'Day', 'Hour', 'Minute','Amount',
             'User', 'Card','Use Chip',
             'Merchant Name', 'Merchant City', 'Merchant State', 'Zip', 'MCC',
             'Bad CVV', 'Bad Card Number', 'Bad Expiration', 'Bad PIN',
             'Bad Zipcode', 'Insufficient Balance', 'Regular', 'Technical Glitch',
             'Is Fraud?'
           ]]
    df.columns =['Transaction-Time', 'Year', 'Month', 'Day', 'Hour', 'Minute',
                 'Amount',
                 'User', 'Card','Use-Chip',
                 'Merchant-Name', 'Merchant-City', 'Merchant-State', 'Zip', 'MCC',
                 'Bad-CVV', 'Bad-Card-Number', 'Bad-Expiration', 'Bad-PIN','Bad-Zipcode',
                 'Insufficient-Balance', 'Regular', 'Technical-Glitch',
                 'Fraud']
    print("Done!")
    return df


start = time.time()
dfclean = clean_data(data)
dfclean.to_csv("data/card_transaction.clean.csv",index=False)
end = time.time()
running_time = end - start
print(f"Completed in {round(running_time, 3)} seconds")