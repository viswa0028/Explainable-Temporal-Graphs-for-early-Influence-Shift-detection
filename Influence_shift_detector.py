import pickle
import numpy as np

class InfluenceShiftDetector:
    """
    Δ_u^t = s_u^t − s_u^{t-1}
    Classify as Rising / Falling / Stable using threshold τ
    """
    def __init__(self, tau_threshold: float = 0.15):
        self.tau = tau_threshold

    def run(self, scores_per_snap):
        print("Detecting influence shifts...")
        shifts_per_snap = []

        for t in range(1, len(scores_per_snap)):
            prev = scores_per_snap[t - 1]
            curr = scores_per_snap[t]
            shifts = {}
            for nid, s_now in curr.items():
                s_prev = prev.get(nid, 0.0)
                delta  = s_now - s_prev
                if delta >  self.tau:
                    label = "Rising"
                elif delta < -self.tau:
                    label = "Falling"
                else:
                    label = "Stable"
                shifts[nid] = {"delta": delta, "label": label, "score": s_now}
            shifts_per_snap.append(shifts)

        rising_counts  = [sum(1 for v in s.values() if v["label"]=="Rising")  for s in shifts_per_snap]
        falling_counts = [sum(1 for v in s.values() if v["label"]=="Falling") for s in shifts_per_snap]
        print(f"Avg rising/snapshot:  {np.mean(rising_counts):.1f}  |  "
              f"Avg falling/snapshot: {np.mean(falling_counts):.1f}")
        return shifts_per_snap


if __name__ == "__main__":
    data_dir = "/Users/viswa/Desktop/PythonProject/Sna/data"

    print("Loading pre-computed data...")
    with open(f"{data_dir}/daily_graphs.pkl", "rb") as f:
        daily_graphs = pickle.load(f)
    
    with open(f"{data_dir}/hybrid_scores.pkl", "rb") as f:
        hybrid_scores = pickle.load(f)
        
    dates = sorted(daily_graphs.keys())

    # Recreate node mapping to construct scores_per_snap correctly
    all_nodes = set()
    for d in dates:
        all_nodes.update(daily_graphs[d].nodes())
    all_nodes = sorted(list(all_nodes))
    node2idx = {n: i for i, n in enumerate(all_nodes)}

    # Convert DataFrame scores to dictionaries per snapshot
    scores_per_snap = []
    for d in dates:
        df = hybrid_scores[d]
        scores = {}
        for _, row in df.iterrows():
            scores[node2idx[row["node"]]] = row["hybrid_score"]
        scores_per_snap.append(scores)

    # Initialize and run the detector
    detector = InfluenceShiftDetector(tau_threshold=0.15)
    shifts_per_snap = detector.run(scores_per_snap)

    # Save the shifts output
    shifts_path = f"{data_dir}/shifts_per_snap.pkl"
    with open(shifts_path, "wb") as f:
        pickle.dump(shifts_per_snap, f)
    print(f"Shift results saved to {shifts_path}")
    
    # Save the detector model instance
    model_path = f"{data_dir}/influence_shift_detector.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(detector, f)
    print(f"InfluenceShiftDetector model saved to {model_path}")
