import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn import  metrics
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

##################### Reading Data #########################
mac = True
if mac:
    data_path = "data/transactions"
    df = pd.read_csv(data_path,index_col =0)
else :
    data_path = "data/transactions.txt"
    df = pd.read_json(data_path, lines=True)
#############################################################


