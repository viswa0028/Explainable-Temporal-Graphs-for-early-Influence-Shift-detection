import os
import numpy as np
import pandas as pd
import networkx as nx
import warnings
import logging
import pickle
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("HiggsPipeline")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CFG = dict(
    data_dir        = "/Users/viswa/Desktop/PythonProject/Sna/data",
    output_dir      = "./higgs_outputs",
    lstm_hidden     = 64,
    lstm_layers     = 2,
    tau_threshold   = 0.15,       # influence shift threshold
    pred_seq_len    = 5,          # LSTM look-back window
    top_k_explain   = 10,
    seed            = 42,
)

torch.manual_seed(CFG["seed"])
np.random.seed(CFG["seed"])

os.makedirs(CFG["data_dir"],   exist_ok=True)
os.makedirs(CFG["output_dir"], exist_ok=True)


class Visualizer:
    def __init__(self, cfg: dict):
        self.out = cfg["output_dir"]

    def plot_dashboard(self, snapshots, scores_per_snap, shifts_per_snap,
                       predictions, explain, top_nodes):
        log.info("Generating visualization dashboard …")

        snap_times = [G["graph"].graph["t_start"] for G in snapshots[1:]]

        fig = plt.figure(figsize=(20, 16), facecolor="#0d1117")
        fig.suptitle(
            "Explainable Temporal Graph — Higgs Twitter Influence Dashboard",
            fontsize=18, color="white", fontweight="bold", y=0.98
        )
        gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_facecolor("#161b22")
        colors = ["#58a6ff", "#3fb950", "#ff7b72", "#d2a8ff", "#ffa657"]
        show_nodes = list(top_nodes)[:5]
        for i, nid in enumerate(show_nodes):
            series = [s.get(nid, np.nan) for s in scores_per_snap]
            ax1.plot(series, color=colors[i % len(colors)],
                     linewidth=1.8, label=f"Node {nid}")
        ax1.set_title("Influence Score Over Snapshots", color="white")
        ax1.set_xlabel("Snapshot", color="#8b949e")
        ax1.set_ylabel("Influence Score", color="#8b949e")
        ax1.tick_params(colors="#8b949e")
        ax1.legend(fontsize=8, facecolor="#161b22", labelcolor="white")
        for spine in ax1.spines.values():
            spine.set_edgecolor("#30363d")
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_facecolor("#161b22")
        r = [sum(1 for v in s.values() if v["label"]=="Rising")  for s in shifts_per_snap]
        f = [sum(1 for v in s.values() if v["label"]=="Falling") for s in shifts_per_snap]
        st= [sum(1 for v in s.values() if v["label"]=="Stable")  for s in shifts_per_snap]
        xs = range(len(shifts_per_snap))
        ax2.stackplot(xs, r, f, st,
                      labels=["Rising", "Falling", "Stable"],
                      colors=["#3fb950", "#ff7b72", "#8b949e"], alpha=0.85)
        ax2.set_title("Shift Classification", color="white")
        ax2.set_xlabel("Snapshot", color="#8b949e")
        ax2.tick_params(colors="#8b949e")
        ax2.legend(fontsize=8, facecolor="#161b22", labelcolor="white")
        for spine in ax2.spines.values():
            spine.set_edgecolor("#30363d")

        ax3 = fig.add_subplot(gs[1, :2])
        ax3.set_facecolor("#161b22")
        top10 = list(top_nodes)[:10]
        delta_mat = []
        for nid in top10:
            row = [sh.get(nid, {}).get("delta", 0.0) for sh in shifts_per_snap]
            delta_mat.append(row)
        delta_mat = np.array(delta_mat)
        im = ax3.imshow(delta_mat, aspect="auto", cmap="RdYlGn",
                        vmin=-0.5, vmax=0.5)
        ax3.set_yticks(range(len(top10)))
        ax3.set_yticklabels([f"N{n}" for n in top10], color="#8b949e", fontsize=8)
        ax3.set_xlabel("Snapshot", color="#8b949e")
        ax3.set_title("Influence Δ Heatmap (top nodes)", color="white")
        ax3.tick_params(colors="#8b949e")
        plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.01).ax.yaxis.set_tick_params(color="#8b949e")
        for spine in ax3.spines.values():
            spine.set_edgecolor("#30363d")

        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_facecolor("#161b22")
        if top_nodes:
            nid    = list(top_nodes)[0]
            actual = [s.get(nid, 0.0) for s in scores_per_snap]
            pred_v = predictions.get(nid, None)
            ax4.plot(actual, color="#58a6ff", linewidth=1.8, label="Actual")
            if pred_v is not None:
                ax4.scatter(len(actual), pred_v, color="#ffa657",
                            zorder=5, s=80, label=f"Pred t+1")
                ax4.axvline(len(actual)-1, color="#30363d", linestyle="--", linewidth=0.8)
            ax4.set_title(f"Prediction — Node {nid}", color="white")
            ax4.set_xlabel("Snapshot", color="#8b949e")
            ax4.tick_params(colors="#8b949e")
            ax4.legend(fontsize=8, facecolor="#161b22", labelcolor="white")
        for spine in ax4.spines.values():
            spine.set_edgecolor("#30363d")

        ax5 = fig.add_subplot(gs[2, 0])
        ax5.set_facecolor("#0d1117")
        sem    = explain["semantic"]
        labels = list(sem.keys())
        vals   = list(sem.values())
        wedge_colors = ["#58a6ff", "#3fb950", "#ff7b72"]
        wedges, texts, autotexts = ax5.pie(
            vals, labels=labels, autopct="%1.1f%%",
            colors=wedge_colors[:len(vals)],
            textprops={"color": "white", "fontsize": 9}
        )
        ax5.set_title("Interaction Type Distribution\n(Semantic Explainability)",
                      color="white", fontsize=9)


        ax6 = fig.add_subplot(gs[2, 1])
        ax6.set_facecolor("#161b22")
        temp_exp = explain["temporal"]
        valid    = {k: v for k, v in temp_exp.items() if v is not None}
        if valid:
            nids_sorted = sorted(valid, key=lambda x: valid[x])
            onset_vals  = [valid[n] for n in nids_sorted]
            bar_colors  = ["#3fb950" if v < np.median(onset_vals) else "#ff7b72"
                           for v in onset_vals]
            ax6.barh([f"N{n}" for n in nids_sorted], onset_vals,
                     color=bar_colors, edgecolor="#0d1117")
            ax6.set_title("Shift Onset Snapshot\n(Temporal Explainability)",
                          color="white", fontsize=9)
            ax6.set_xlabel("Snapshot index", color="#8b949e")
            ax6.tick_params(colors="#8b949e")
        for spine in ax6.spines.values():
            spine.set_edgecolor("#30363d")


        ax7 = fig.add_subplot(gs[2, 2])
        ax7.set_facecolor("#161b22")
        struct = explain["structural"]
        if struct:
            edge_labels = [f"{u}→{v}" for u, v, _, _ in struct[:8]]
            edge_scores = [sc for _, _, sc, _ in struct[:8]]
            y_pos = range(len(edge_labels))
            ax7.barh(y_pos, edge_scores, color="#d2a8ff", edgecolor="#0d1117")
            ax7.set_yticks(list(y_pos))
            ax7.set_yticklabels(edge_labels, color="#8b949e", fontsize=7)
            ax7.set_title("Top Structural Edges\n(Structural Explainability)",
                          color="white", fontsize=9)
            ax7.set_xlabel("Influence Score", color="#8b949e")
            ax7.tick_params(colors="#8b949e")
        for spine in ax7.spines.values():
            spine.set_edgecolor("#30363d")

        out_path = os.path.join(self.out, "influence_dashboard.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor="#0d1117")
        plt.close()
        log.info(f"Dashboard saved → {out_path}")
        return out_path

    def export_csv(self, scores_per_snap, shifts_per_snap, top_nodes):
        rows = []
        for t, (scores, shifts) in enumerate(zip(scores_per_snap, shifts_per_snap)):
            for nid in top_nodes:
                rows.append({
                    "snapshot": t,
                    "node_id":  nid,
                    "score":    scores.get(nid, 0.0),
                    "delta":    shifts.get(nid, {}).get("delta", 0.0),
                    "label":    shifts.get(nid, {}).get("label", "N/A"),
                })
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.out, "influence_scores.csv")
        df.to_csv(csv_path, index=False)
        log.info(f"Scores CSV saved → {csv_path}")
        return csv_path

def run_pipeline():
    log.info("=" * 65)
    log.info("  Explainable Temporal Graph Learning Framework")
    log.info("  Dataset: Higgs Twitter (SNAP Stanford)")
    log.info("=" * 65)

    data_dir = CFG["data_dir"]

    log.info("\n[Stage 1-6] Loading pre-computed data …")
    
    with open(f"{data_dir}/daily_graphs.pkl", "rb") as f:
        daily_graphs = pickle.load(f)
    
    with open(f"{data_dir}/features.pkl", "rb") as f:
        features_by_day = pickle.load(f)
        
    with open(f"{data_dir}/hybrid_scores.pkl", "rb") as f:
        hybrid_scores = pickle.load(f)
        
    retweet_time = pd.read_csv(f"{data_dir}/retweet_time.csv")
    temporal_embeddings = torch.load(f"{data_dir}/temporal_embeddings.pt")
        
    dates = sorted(daily_graphs.keys())

    all_nodes = set()
    for d in dates:
        all_nodes.update(daily_graphs[d].nodes())
    all_nodes = sorted(list(all_nodes))
    node2idx = {n: i for i, n in enumerate(all_nodes)}
    n_nodes = len(node2idx)

    enriched_snaps = []
    scores_per_snap = []
    
    for d in dates:
        G = daily_graphs[d]
        feat_df = features_by_day[d]
        score_df = hybrid_scores[d]
        
        G_idx = nx.DiGraph()
        G_idx.graph["t_start"] = pd.to_datetime(d)
        for u, v in G.edges():
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

    if not scores_per_snap:
        log.error("No scores computed. Exiting.")
        return

    # pick top nodes by average score across snapshots
    agg_scores = defaultdict(float)
    for sc in scores_per_snap:
        for nid, v in sc.items():
            agg_scores[nid] += v
    top_nodes = sorted(agg_scores, key=lambda x: -agg_scores[x])[:CFG["top_k_explain"]]

    # ── Stage 7-9 ─────────────────────────────────────────────────────────
    log.info("\n[Stage 7-9] Loading models and outputs …")
    
    with open(f"{data_dir}/influence_shift_detector.pkl", "rb") as f:
        detector_model = pickle.load(f)
    with open(f"{data_dir}/shifts_per_snap.pkl", "rb") as f:
        shifts_per_snap = pickle.load(f)

    with open(f"{data_dir}/predictions.pkl", "rb") as f:
        predictions = pickle.load(f)

    with open(f"{data_dir}/explainability_module.pkl", "rb") as f:
        explainer_model = pickle.load(f)
    with open(f"{data_dir}/explainability.pkl", "rb") as f:
        explain = pickle.load(f)


    log.info("\n[Stage 10] Visualization & insights …")
    viz      = Visualizer(CFG)
    dash_path = viz.plot_dashboard(
        enriched_snaps, scores_per_snap, shifts_per_snap,
        predictions, explain, top_nodes
    )
    csv_path = viz.export_csv(scores_per_snap, shifts_per_snap, top_nodes)

    log.info("\n" + "=" * 65)
    log.info("  PIPELINE COMPLETE")
    log.info(f"  Nodes processed      : {n_nodes:,}")
    log.info(f"  Temporal snapshots   : {len(dates)}")
    log.info(f"  Top rising nodes     : "
             f"{[n for n in top_nodes if shifts_per_snap and shifts_per_snap[-1].get(n,{}).get('label')=='Rising'][:5]}")
    log.info(f"  Dashboard            : {dash_path}")
    log.info(f"  Scores CSV           : {csv_path}")
    log.info("=" * 65)

    return {
        "n_nodes":        n_nodes,
        "snapshots":      len(dates),
        "scores":         scores_per_snap,
        "shifts":         shifts_per_snap,
        "predictions":    predictions,
        "explain":        explain,
        "top_nodes":      top_nodes,
        "dashboard_path": dash_path,
        "csv_path":       csv_path,
    }

if __name__ == "__main__":
    results = run_pipeline()