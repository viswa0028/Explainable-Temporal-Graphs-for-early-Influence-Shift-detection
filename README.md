# Explainable-Temporal-Graphs-for-early-Influence-Shift-detection

This project implements a comprehensive pipeline for analyzing dynamic social networks, detecting shifts in node influence over time, tracking community evolution, and providing explainable forecasts. It is specifically tailored for dynamic network datasets (like the Higgs Twitter dataset) and includes a rich, interactive dashboard built with Streamlit.

## 🚀 Features

- **Dynamic Network Analysis**: Process network snapshots over time to understand evolving structures.
- **Hybrid Influence Scoring**: Computes robust node influence scores by combining multiple centrality metrics (degree, betweenness, PageRank, etc.).
- **Community Tracking**: Uses the Louvain method to track the birth, death, merging, and splitting of network communities across time steps.
- **Proactive Shift Prediction**: Utilizes a custom LSTM model with Temporal Attention (built in PyTorch) to forecast future influence scores and proactively alert on rising or falling nodes.
- **Explainability**: Integrates SHAP values to explain the predictions of the influence shift detector, providing transparency into *why* a node's influence is predicted to change.
- **Interactive Dashboard**: A comprehensive Streamlit dashboard for visualizing network metrics, tracking specific nodes, exploring community evolution, and viewing prediction explanations.

## 📁 Project Structure

- `notebooks/`: Contains the core analytical modules and the Streamlit dashboard.
  - `Influence_shift_detector.py` / `Influence_shift_detector_static.py`: Core logic for community tracking and LSTM-based influence forecasting.
  - `dasboardfinal.py`: The Streamlit application for visualizing the analysis.
  - `embeddings.py`: Generates node embeddings (e.g., using Node2Vec or Graph Neural Networks).
  - `hybrid_score.py`: Computes the hybrid centrality scores.
  - `ExplainabilityModule.py`: Logic for generating SHAP-based explanations.
  - `feature_extraction.py` & `preprocessing.py`: Utilities for preparing the network data.
  - `validation.py` & `compare_validation.py` & `plot_comparison.py`: Scripts for validating model performance and generating comparison plots.
- `data/`: Directory for storing input datasets (e.g., edge lists), intermediate pickle files (like `daily_graphs.pkl`), and model outputs.
- `requirements.txt`: Python dependencies for the project.

## 🛠 Installation

1. **Clone the repository** (if applicable) or navigate to the project directory:
   ```bash
   cd /path/to/Sna
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install the dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This project relies on PyTorch and PyTorch Geometric. Ensure you have the appropriate versions installed for your hardware (CPU or CUDA).*

## 💻 Usage

### 1. Run the Backend Pipeline
To process the data, track communities, train the LSTM predictor, and generate predictions, run the main analysis script:

```bash
python notebooks/Influence_shift_detector.py
```
This will process the data in the `data/` directory and output several `.pkl` files containing graphs, hybrid scores, community history, and proactive alerts.

### 2. Launch the Dashboard
Once the data is processed, you can launch the interactive Streamlit dashboard to visualize the results:

```bash
streamlit run notebooks/dasboardfinal.py
```
This will open the dashboard in your default web browser (usually at `http://localhost:8501`), where you can explore the Influence Shift Detector interface.

## 📊 Technologies Used

- **Data Processing & Machine Learning**: `pandas`, `numpy`, `scikit-learn`
- **Network Analysis**: `networkx`, `torch-geometric`, `pyvis`
- **Deep Learning**: `torch`, `torchvision`, `torchaudio` (LSTM with Attention)
- **Explainability**: `shap`
- **Visualization & UI**: `matplotlib`, `streamlit`
