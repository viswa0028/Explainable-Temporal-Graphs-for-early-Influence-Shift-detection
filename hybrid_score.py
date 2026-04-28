import pickle
import torch
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# -----------------------------------
# Load features and embeddings
# -----------------------------------
with open("/Users/viswa/Desktop/PythonProject/Sna/data/features.pkl", "rb") as f:
    features_by_day = pickle.load(f)

temporal_embeddings = torch.load("/Users/viswa/Desktop/PythonProject/Sna/data/temporal_embeddings.pt")
# shape: [num_nodes, time_steps, hidden_dim]

with open("/Users/viswa/Desktop/PythonProject/Sna/data/daily_graphs.pkl", "rb") as f:
    daily_graphs = pickle.load(f)

dates = sorted(features_by_day.keys())

# rebuild node mapping
all_nodes = set()
for d in dates:
    all_nodes.update(daily_graphs[d].nodes())

all_nodes = sorted(list(all_nodes))
node_to_idx = {n:i for i,n in enumerate(all_nodes)}

# -----------------------------------
# Parameters
# -----------------------------------
lambda_val = 0.5

hybrid_scores = {}

# -----------------------------------
# Compute per day scores
# -----------------------------------
for t, day in enumerate(dates):

    df = features_by_day[day].copy()

    # Static Score
    static_cols = ["indegree", "outdegree", "pagerank", "betweenness", "clustering"]
    scaler = MinMaxScaler()
    static_scaled = scaler.fit_transform(df[static_cols])
    df["static_score"] = static_scaled.mean(axis=1)

    # Embedding Score
    emb_scores = []
    for node in df["node"]:
        idx = node_to_idx[node]
        emb = temporal_embeddings[idx, t, :]   # node embedding at day t
        emb_score = emb.mean().item()
        emb_scores.append(emb_score)

    df["embedding_score"] = emb_scores

    # Normalize embedding score
    df["embedding_score"] = MinMaxScaler().fit_transform(
        df[["embedding_score"]]
    )

    # Final Hybrid Score
    df["hybrid_score"] = (
        lambda_val * df["static_score"] +
        (1 - lambda_val) * df["embedding_score"]
    )

    hybrid_scores[day] = df[
        ["node", "static_score", "embedding_score", "hybrid_score"]
    ]

with open("/Users/viswa/Desktop/PythonProject/Sna/data/hybrid_scores.pkl", "wb") as f:
    pickle.dump(hybrid_scores, f)

print("Hybrid scores saved!")

print(hybrid_scores[dates[0]].head())