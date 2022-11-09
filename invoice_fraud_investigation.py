import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from faker import Faker
import  numpy as np
import random



# Profiles
fake = Faker(['en_US',"it_IT"])
# for _ in range(5):
#     print( pd.Series( [fake.company(),fake.iban(),fake.pricetag(),fake.date_between(-90), fake.local_latlng("IT")]) )

# Problem statements
#
# quale di queste compagnie è una shell company coinvolta nei flussi di money laundering con modalità invoice fraud
#

# creazione dei nodi
n_boss = 3
n_shell = 20
n_clean = 40
it= fake["it_IT"]
us = fake["en_US"]
compagnie_boss = [us.company() for _ in range(n_boss)]
compagnie_shell = [it.company() for _ in range(n_shell)]
compagnie_clean = [it.company() for _ in range(n_clean)]
compagnie_hybrid = np.random.choice(compagnie_shell,3).tolist()

# creazione link tra aziende coinvolte nella frode. 2 trx dello stesso amount (A,B,50);(B,A,50)
def create_invoice_fraud(boss,shell):
    # Gli importi delle frodi sono sempre numeri tondi a cifra 0
    return (random.choice(boss),random.choice(shell))

def create_invoice_clean(compagnie_clean,compagnie_hybrid):
    # Amount,Client,Date
    A = compagnie_clean
    B = compagnie_clean
    return (random.choice(A),random.choice(B))

def clean_invoice_to_bad(compagnie_clean,compagnie_boss):
    # Amount,Client,Date
    A = compagnie_clean
    B = compagnie_boss
    return (random.choice(A), random.choice(B))


# creazione della frode, scambio a stessa cifra tra boss e shell
trx_invoice_gone = [create_invoice_fraud(compagnie_boss,compagnie_shell) for _ in range(300)]
trx_invoice_back  = [trx_invoice_gone[i][::-1] for i in range(len(trx_invoice_gone))]
invoice_frauds = trx_invoice_gone + trx_invoice_back
# creazione transazioni normali tra compagnie minori, nota che le USA sono sempre boss e coinvolte nella frode
invoice_clean = [create_invoice_clean(compagnie_clean,compagnie_shell) for _ in range(100)]
invoice_clean_to_boss_clean = [clean_invoice_to_bad(compagnie_clean,compagnie_shell) for _ in range(50)]
total_trx = invoice_frauds+invoice_clean+invoice_clean_to_boss_clean

# creazione amount frodi
amount_blue_print = [i*100 for i in range(1,10)]
amount_frauds = [random.choice(amount_blue_print) for _ in range(len(invoice_frauds))]

#creazione amount clean
def random_round(x):
    if np.random.choice([0, 1], 1, p = [0.9, 0.1]).item() == 1:
        return 200
    else:
        return  x

amount_clean = list(map(random_round,np.random.normal(500,90,len(invoice_clean+invoice_clean_to_boss_clean)).round(2).tolist()))

# df transazioni
nodes = pd.DataFrame(total_trx,columns = ["from","to"])
df = nodes.copy()
df["amount"] = amount_frauds + amount_clean
df["label"] = ["invoice_fraud"  if trx in invoice_frauds else "clean" for trx in total_trx]

# creazione grafo transazioni
G = nx.MultiDiGraph()
G = nx.from_edgelist(total_trx,G)

# network measures
node_degree = nx.degree(G)
in_degree = G.in_degree()
out_degree = G.out_degree()

df["node_degree_from"] = df["from"].map(node_degree)
df["node_degree_to"] = df["to"].map(node_degree)
df["in_degree_from"] = df["from"].map(in_degree)
df["in_degree_to"] = df["to"].map(in_degree)
df["out_degree_from"] = df["from"].map(out_degree)
df["out_degree_to"] = df["to"].map(out_degree)

def closed_walk_in_network(A):
    counter = 0
    destinations = [list(G.out_edges(A))[idx][-1] for idx in range(len(list(G.out_edges(A))))]
    for d in destinations:
        if A in [list(G.out_edges(d))[idx][-1] for idx in range(len(list(G.out_edges(d))))]:
            counter+=1
    return counter


number_of_closed_walk = pd.Series([closed_walk_in_network(A) for A in G.nodes], index = G.nodes)
df["closed_walk_from"] = df["from"].map(number_of_closed_walk)
df["closed_walk_to"] = df["to"].map(number_of_closed_walk)


# Visualization
nsize = [(s+1)*10 for s in number_of_closed_walk.values ]
edgew = [w/100 for w in df["closed_walk_from"].values ]
edge_col = [ "red" if i in compagnie_boss  else "black" for i in df["from"]  ]
nodecolors =  [G.degree(v) for v in G] #["red" if node in compagnie_boss else "yellow" for node in G  ]
plt.figure(figsize=(10,10))
pos=nx.spiral_layout(G)
nx.draw_networkx(G,pos,with_labels=True,node_color = nodecolors, node_size = nsize,cmap = plt.cm.Blues,width=edgew,edge_color = edge_col)
ax = plt.gca()
plt.axis('off')
plt.tight_layout();