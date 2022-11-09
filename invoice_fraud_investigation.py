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
n_boss = 10
n_shell = 50
n_clean = 40
it= fake["it_IT"]
us = fake["en_US"]
compagnie_boss = [us.company() for _ in range(n_boss)]
compagnie_shell = [it.company() for _ in range(n_shell)]
compagnie_clean = [it.company() for _ in range(n_clean)]

# creazione link tra aziende coinvolte nella frode. 2 trx dello stesso amount (A,B,50);(B,A,50)
def create_invoice_fraud(boss,shell):
    # Gli importi delle frodi sono sempre numeri tondi a cifra 0
    amount = [i*100 for i in range(1,10)]
    return (random.choice(boss),random.choice(shell))

def create_invoice_clean(compagnie_clean,compagnie_shell):
    # Amount,Client,Date
    A = compagnie_clean
    B = compagnie_clean
    amount = [i*100 for i in range(1,10)]
    return (random.choice(A),random.choice(B))

# creazione della frode, scambio a stessa cifra tra boss e shell
trx_invoice_gone = [create_invoice_fraud(compagnie_boss,compagnie_shell) for _ in range(200)]
trx_invoice_back  = [trx_invoice_gone[i][::-1] for i in range(len(trx_invoice_gone))]
invoice_frauds = trx_invoice_gone + trx_invoice_back
# creazione transazioni normali tra compagnie minori, nota che le USA sono sempre boss e coinvolte nella frode
invoice_clean = [create_invoice_clean(compagnie_clean,compagnie_shell) for _ in range(100)]
total_trx = invoice_frauds+invoice_clean

# creazione amount frodi
amount_blue_print = [i*100 for i in range(1,10)]
amount_frauds = [random.choice(amount_blue_print) for _ in range(len(invoice_frauds))]

#creazione amount clean
def random_round(x):
    if np.random.choice([0, 1], 1, p = [0.9, 0.1]).item() == 1:
        return 200
    else:
        return  x

amount_clean = list(map(random_round,np.random.normal(500,90,len(invoice_clean)).round(2).tolist()))



# creazione grafo transazioni
G = nx.DiGraph()
G = nx.from_edgelist(total_trx,G)

# df transazioni
nodes = pd.DataFrame(total_trx,columns = ["from","to"])
df = nodes.copy()
df["amount"] = amount_frauds + amount_clean
df["label"] = ["invoice_fraud"  if trx in invoice_frauds else "clean" for trx in total_trx]

# network measures
node_degree = nx.degree(G)
in_degree = G.in_degree()
out_degree = G.out_degree()
eigenvector_centrality = nx.eigenvector_centrality(G)

df["node_degree_from"] = df["from"].map(node_degree)
df["node_degree_to"] = df["to"].map(node_degree)
df["in_degree_from"] = df["from"].map(in_degree)
df["in_degree_to"] = df["to"].map(in_degree)
df["out_degree_from"] = df["from"].map(out_degree)
df["out_degree_to"] = df["to"].map(out_degree)
df["eigenvector_centrality_from"] = df["from"].map(eigenvector_centrality)
df["eigenvector_centrality_to"] = df["to"].map(eigenvector_centrality)

# totale_compagnie = compagnie_clean+compagnie_shell+compagnie_boss
# for i in totale_compagnie:
#     print("-"*50)
#     print(i)
#     print(f"Eterogeneità pagamenti ad altra compagnia: {df.loc[df['from']== i,'to'].nunique()}")
#     print(f"Eterogeneità pagamenti da altra compagnia: {df.loc[df['to'] == i, 'from'].nunique()}")

# Visualization
nsize = [s*100 for s in dict(node_degree).values() ]
edgew = [w/10 for w in df["in_degree_from"].values ]
nodecolors =  [G.degree(v) for v in G] #["red" if node in compagnie_boss else "yellow" for node in G  ]
plt.figure(figsize=(100,10))
pos=nx.spiral_layout(G)
nx.draw_networkx(G,pos,with_labels=True,node_color = nodecolors, node_size = nsize,width = edgew,cmap = plt.cm.Blues)
ax = plt.gca()
ax.collections[0].set_edgecolor("#696969")
plt.axis('off')
plt.tight_layout();




# creazione grafo relazioni
Gno = nx.Graph()
Gno = nx.from_edgelist(total_trx,G)


def check_circuit(A,B,G):
    """
    Quante volte i nodi A e B creano un circuito semplice
    Definizione di Circuito semplice : (AB,BA)
    Args :
        * A nodo origine
        * B nodo destinazione
        * G grafo orientato
    Returns a tuple (A,B,n) dove n è il numero di circuiti semplici completati
    """
    # a circuit is a special case of cycles
    circuit = list(nx.find_cycle(G, source=compagnie_boss[0], orientation="ignore"))
    return (A,B,n)


for node in G.nodes():
    print(node)
