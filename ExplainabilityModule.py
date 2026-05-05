import pickle
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict

class ExplainabilityModule:
    """
    Structural:  top-K influential edges (by attention proxy = edge weight)
    Temporal:    snapshot where shift onset occurred
    Semantic:    interaction-type distribution driving the shift
    """
    def __init__(self, top_k_explain: int = 10):
        self.k = top_k_explain

    def structural(self, enriched_snaps, scores_per_snap, top_nodes):
        """Top-K edges contributing to high-influence nodes."""
        print("Explainability — structural...")
        important_edges = []
        if not scores_per_snap:
            return important_edges
        last_scores = scores_per_snap[-1]
        top_set     = set(top_nodes)
        G_last      = enriched_snaps[-1]["graph"]
        for u, v, d in G_last.edges(data=True):
            if u in top_set or v in top_set:
                score_uv = (last_scores.get(u, 0) + last_scores.get(v, 0)) / 2
                important_edges.append((u, v, score_uv, d.get("itype", 0)))
        important_edges.sort(key=lambda x: -x[2])
        return important_edges[:self.k]

    def temporal(self, shifts_per_snap, node_id):
        """Find snapshot index where node first became Rising."""
        for t, shifts in enumerate(shifts_per_snap):
            if shifts.get(node_id, {}).get("label") == "Rising":
                return t
        return None

    def semantic(self, enriched_snaps, top_nodes):
        """Interaction-type distribution for top nodes."""
        itype_labels = {0: "Retweet", 1: "Mention", 2: "Reply"}
        counts = defaultdict(int)
        for snap in enriched_snaps:
            G = snap["graph"]
            for u, v, d in G.edges(data=True):
                if u in top_nodes or v in top_nodes:
                    counts[itype_labels.get(d.get("itype", 0), "Unknown")] += 1
        total = sum(counts.values()) or 1
        return {k: v / total for k, v in counts.items()}

    def run(self, enriched_snaps, scores_per_snap, shifts_per_snap, top_nodes, node2idx):
        print("Running full explainability analysis...")
        top_n = list(top_nodes)[:self.k]

        struct   = self.structural(enriched_snaps, scores_per_snap, set(top_n))
        temporal = {nid: self.temporal(shifts_per_snap, nid) for nid in top_n}
        semantic = self.semantic(enriched_snaps, set(top_n))

        idx2node = {v: k for k, v in node2idx.items()}
        print(f"Structural — top {len(struct)} influential edges found")
        print(f"Semantic   — {semantic}")
        
        return {
            "structural": struct,
            "temporal":   temporal,
            "semantic":   semantic,
            "idx2node":   idx2node,
        }

if __name__ == "__main__":
    data_dir = "/Users/viswa/Desktop/PythonProject/Sna/data"
    
    print("Loading data...")
    with open(f"{data_dir}/daily_graphs.pkl", "rb") as f:
        daily_graphs = pickle.load(f)
    with open(f"{data_dir}/features.pkl", "rb") as f:
        features_by_day = pickle.load(f)
    with open(f"{data_dir}/hybrid_scores.pkl", "rb") as f:
        hybrid_scores = pickle.load(f)
    with open(f"{data_dir}/shifts_per_snap.pkl", "rb") as f:
        shifts_per_snap = pickle.load(f)

    dates = sorted(daily_graphs.keys())
    all_nodes = set()
    for d in dates:
        all_nodes.update(daily_graphs[d].nodes())
    all_nodes = sorted(list(all_nodes))
    node2idx = {n: i for i, n in enumerate(all_nodes)}

    # Reconstruct enriched_snaps and scores_per_snap
    enriched_snaps = []
    scores_per_snap = []
    for d in dates:
        G = daily_graphs[d]
        feat_df = features_by_day[d]
        score_df = hybrid_scores[d]
        
        G_idx = nx.DiGraph()
        G_idx.graph["t_start"] = pd.to_datetime(d)
        for u, v in G.edges():
            # For simplicity assuming itype=0 (RT) based on data scope
            G_idx.add_edge(node2idx[u], node2idx[v], itype=0)
            
        node_feats = {}
        for _, row in feat_df.iterrows():
            n = row["node"]
            feat_vec = np.array([row["indegree"], row["pagerank"], row["betweenness"]], dtype=np.float32)
            node_feats[node2idx[n]] = feat_vec
            
        enriched_snaps.append({"graph": G_idx, "node_feats": node_feats, "edge_feats": None})
        
        scores = {}
        for _, row in score_df.iterrows():
            scores[node2idx[row["node"]]] = row["hybrid_score"]
        scores_per_snap.append(scores)

    # Recompute top nodes
    top_k_explain = 10
    agg_scores = defaultdict(float)
    for sc in scores_per_snap:
        for nid, v in sc.items():
            agg_scores[nid] += v
    top_nodes = sorted(agg_scores, key=lambda x: -agg_scores[x])[:top_k_explain]

    # Run explainability module
    explainer = ExplainabilityModule(top_k_explain=top_k_explain)
    explain = explainer.run(enriched_snaps, scores_per_snap, shifts_per_snap, top_nodes, node2idx)

    # Save outputs and model
    with open(f"{data_dir}/explainability.pkl", "wb") as f:
        pickle.dump(explain, f)
        
    with open(f"{data_dir}/explainability_module.pkl", "wb") as f:
        pickle.dump(explainer, f)
    
    print("Explainability analysis complete and results saved.")
