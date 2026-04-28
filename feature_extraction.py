import pickle
import pandas as pd
import networkx as nx

with open("/Users/viswa/Desktop/PythonProject/Sna/data/daily_graphs.pkl", "rb") as f:
    daily_graphs = pickle.load(f)
features_by_day = {}

prev_degree = {}

for day, G in daily_graphs.items():
    
    indeg = dict(G.in_degree())
    outdeg = dict(G.out_degree())
    pagerank = nx.pagerank(G)
    bet = nx.betweenness_centrality(G)
    cluster = nx.clustering(G.to_undirected())

    rows = []

    for node in G.nodes():
        total_deg = indeg.get(node, 0) + outdeg.get(node, 0)
        old_deg = prev_degree.get(node, 0)
        degree_change = total_deg - old_deg

        rows.append({
            "node": node,
            "indegree": indeg.get(node, 0),
            "outdegree": outdeg.get(node, 0),
            "pagerank": pagerank.get(node, 0),
            "betweenness": bet.get(node, 0),
            "clustering": cluster.get(node, 0),
            "degree_change": degree_change
        })

        prev_degree[node] = total_deg

    features_by_day[day] = pd.DataFrame(rows)

with open("/Users/viswa/Desktop/PythonProject/Sna/data/features.pkl", "wb") as f:
    pickle.dump(features_by_day, f)