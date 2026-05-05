import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from collections import defaultdict

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class InfluencePredictor(nn.Module):
    def __init__(self, lstm_hidden: int = 64, lstm_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(1, lstm_hidden, lstm_layers, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(lstm_hidden, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class PredictionModule:
    def __init__(self, lstm_hidden: int = 64, lstm_layers: int = 2, pred_seq_len: int = 5):
        self.model = InfluencePredictor(lstm_hidden, lstm_layers).to(DEVICE)
        self.opt   = Adam(self.model.parameters(), lr=1e-3)
        self.seq   = pred_seq_len

    def _build_sequences(self, score_series):
        X, Y = [], []
        for i in range(len(score_series) - self.seq):
            X.append(score_series[i: i + self.seq])
            Y.append(score_series[i + self.seq])
        if not X:
            return None, None
        return (torch.tensor(np.array(X), dtype=torch.float, device=DEVICE).unsqueeze(-1),
                torch.tensor(np.array(Y), dtype=torch.float, device=DEVICE).unsqueeze(-1))

    def run(self, scores_per_snap, top_nodes):
        print("Training prediction module (LSTM)...")
        predictions = {}

        for nid in top_nodes:
            series = [s.get(nid, 0.0) for s in scores_per_snap]
            # Need at least sequence length + 2 entries to train and predict next
            if len(series) < self.seq + 2:
                continue

            # normalise
            s_arr  = np.array(series, dtype=np.float32)
            s_min, s_max = s_arr.min(), s_arr.max()
            denom  = s_max - s_min + 1e-9
            s_norm = (s_arr - s_min) / denom

            X, Y = self._build_sequences(s_norm)
            if X is None:
                continue

            self.model.train()
            for _ in range(15):   # quick fine-tune per node
                self.opt.zero_grad()
                pred = self.model(X)
                loss = F.mse_loss(pred, Y)
                loss.backward()
                self.opt.step()

            self.model.eval()
            with torch.no_grad():
                inp  = torch.tensor(s_norm[-self.seq:], dtype=torch.float, device=DEVICE).unsqueeze(0).unsqueeze(-1)
                pred = self.model(inp).item()
                
            # denormalise
            predictions[nid] = pred * denom + s_min

        print(f"Predictions ready for {len(predictions)} nodes")
        return predictions


if __name__ == "__main__":
    data_dir = "/Users/viswa/Desktop/PythonProject/Sna/data"
    
    print("Loading pre-computed data...")
    with open(f"{data_dir}/hybrid_scores.pkl", "rb") as f:
        hybrid_scores = pickle.load(f)
    with open(f"{data_dir}/daily_graphs.pkl", "rb") as f:
        daily_graphs = pickle.load(f)

    dates = sorted(daily_graphs.keys())
    
    # Reconstruct node indices
    all_nodes = set()
    for d in dates:
        all_nodes.update(daily_graphs[d].nodes())
    all_nodes = sorted(list(all_nodes))
    node2idx = {n: i for i, n in enumerate(all_nodes)}

    # Build scores per snapshot mapped to idx
    scores_per_snap = []
    for d in dates:
        df = hybrid_scores[d]
        scores = {}
        for _, row in df.iterrows():
            scores[node2idx[row["node"]]] = row["hybrid_score"]
        scores_per_snap.append(scores)

    # Determine top nodes
    top_k_explain = 10
    agg_scores = defaultdict(float)
    for sc in scores_per_snap:
        for nid, v in sc.items():
            agg_scores[nid] += v
    top_nodes = sorted(agg_scores, key=lambda x: -agg_scores[x])[:top_k_explain]

    # Initialize and run predictor
    predictor = PredictionModule(lstm_hidden=64, lstm_layers=2, pred_seq_len=5)
    predictions = predictor.run(scores_per_snap, top_nodes)

    # Save output and model
    with open(f"{data_dir}/predictions.pkl", "wb") as f:
        pickle.dump(predictions, f)
        
    torch.save(predictor.model.state_dict(), f"{data_dir}/influence_predictor_model.pth")
    
    print("Prediction module run complete and models saved.")
