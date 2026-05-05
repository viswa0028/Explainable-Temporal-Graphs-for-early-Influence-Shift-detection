# TGNN Dynamic Embeddings

import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------
# Load graphs and features
# ---------------------------------------------------
with open("/Users/viswa/Desktop/PythonProject/Sna/data/daily_graphs.pkl", "rb") as f:
    daily_graphs = pickle.load(f)

with open("/Users/viswa/Desktop/PythonProject/Sna/data/features.pkl", "rb") as f:
    features_by_day = pickle.load(f)

dates = sorted(daily_graphs.keys())

# ---------------------------------------------------
# Build global node mapping
# ---------------------------------------------------
all_nodes = set()
for d in dates:
    all_nodes.update(daily_graphs[d].nodes())

all_nodes = sorted(list(all_nodes))
node_to_idx = {n:i for i,n in enumerate(all_nodes)}
idx_to_node = {i:n for n,i in node_to_idx.items()}

num_nodes = len(all_nodes)
input_dim = 6
hidden_dim = 64

# ---------------------------------------------------
# Convert each day to PyG Data
# ---------------------------------------------------
graph_data = []

for d in dates:
    G = daily_graphs[d]
    feat_df = features_by_day[d]

    x = torch.zeros((num_nodes, input_dim), dtype=torch.float)

    for _, row in feat_df.iterrows():
        idx = node_to_idx[row["node"]]
        x[idx] = torch.tensor([
            row["indegree"],
            row["outdegree"],
            row["pagerank"],
            row["betweenness"],
            row["clustering"],
            row["degree_change"]
        ], dtype=torch.float)

    edges = []
    for u, v in G.edges():
        edges.append([node_to_idx[u], node_to_idx[v]])

    if len(edges) == 0:
        edge_index = torch.empty((2,0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    graph_data.append(data)

# ---------------------------------------------------
# TGNN Model = GCN + GRU
# ---------------------------------------------------
class TGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(TGNN, self).__init__()
        self.gcn1 = GCNConv(in_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, snapshots):
        embeddings = []

        for data in snapshots:
            x = data.x.to(device)
            edge_index = data.edge_index.to(device)

            h = self.gcn1(x, edge_index)
            h = torch.relu(h)

            h = self.gcn2(h, edge_index)
            h = torch.relu(h)

            embeddings.append(h)

        seq = torch.stack(embeddings, dim=1)   # [nodes, time, hidden]
        out, _ = self.gru(seq)
        final_emb = out[:, -1, :]             # last timestep embedding

        return final_emb, out

# ---------------------------------------------------
# Initialize model
# ---------------------------------------------------
model = TGNN(input_dim, hidden_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------------------
# Self-supervised training
# ---------------------------------------------------
for epoch in range(50):
    model.train()
    optimizer.zero_grad()

    final_emb, seq_out = model(graph_data)

    loss = 0
    for t in range(len(graph_data)-1):
        loss += ((seq_out[:, t, :] - seq_out[:, t+1, :])**2).mean()

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# ---------------------------------------------------
# Save embeddings
# ---------------------------------------------------
model.eval()
with torch.no_grad():
    final_emb, seq_out = model(graph_data)

torch.save(final_emb.cpu(), "/Users/viswa/Desktop/PythonProject/Sna/data/final_embeddings.pt")
torch.save(seq_out.cpu(), "/Users/viswa/Desktop/PythonProject/Sna/data/temporal_embeddings.pt")

print("Embeddings saved!")

# ---------------------------------------------------
# Example: embedding of first node
# ---------------------------------------------------
print(final_emb.shape)   # [num_nodes, hidden_dim]
print(final_emb[0])