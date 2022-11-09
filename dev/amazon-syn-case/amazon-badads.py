import pandas as pd
import numpy as np
import networkx as nx




path = "data/Advertisement_AdClickFraud_20k.csv"
amz_data = pd.read_csv(path)


G = nx.from_pandas_edgelist( df= amz_data,source = "ip_address",target = "publisher_id",create_using = nx.DiGraph())
