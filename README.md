# 🛡️ Temporal GNN for Financial Fraud Detection

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

This is a machine learning project dedicated to detecting fraudulent interactions in temporal trust networks using deep learning. By leveraging Temporal Graph Neural Networks (TGNN) equipped with GRU-based state updates (implicit memory), this project models the dynamic evolution of user interactions to identify anomalous and fraudulent behaviors.

## 📊 The Dataset: Bitcoin OTC Trust Network

The project utilizes the **Bitcoin OTC Trust Network** dataset . 

This dataset represents a who-trusts-whom network of people who trade using Bitcoin on a platform called Bitcoin OTC. Since Bitcoin users are anonymous, there is a need to maintain a record of users' reputations to prevent transactions with fraudulent and risky users. 

- **Nodes:** Individual users on the platform.
- **Edges:** Over-the-counter trust ratings given by one user to another.
- **Raw Features:** Timestamps and rating values.
- **Engineered Features:** Historical interaction statistics (counts, ratios, etc.).
- **Fraud Rate:** ~10% of the interactions represent fraudulent or untrustworthy activities.

## 🧠 About the Temporal GNN Model

The core of this project is a custom PyTorch-based **Temporal Graph Neural Network (TGNN)** designed to capture both structural relationships and temporal dynamics. Fraudulent behavior is often hidden in the *timing* and *sequence* of interactions, making static graph models insufficient.

Our `TemporalModel` architecture comprises:
- **Feature Projection:** A linear layer to project raw edge features into a dense vector space.
- **Continual Time Tracking:** Incorporation of temporal ordering and interaction timing to understand bursty vs. sporadic relationships.
- **GRU-based State Updates:** Two parallel `GRUCell` recurrent modules update the interacting "source" (user) and "target" (item) nodes. These cells continuously evolve individual node representations by ingesting historical node embeddings, projected edge features, and interaction timings.
- **MLP Classifier:** A standard Multi-Layer Perceptron (MLP) with sequential dense layers, ReLU activations, and Dropout regularization. It processes the concatenated, continuously-updated embeddings of two interacting nodes to output the final fraud probability score.

## 🛠️ Methodology: How We Proceeded

The system predicts fraudulent interactions (edges) by modeling the temporal dynamics of the graph:

1. **Data Preprocessing & Feature Engineering:** 
   - Loaded and chronologically sorted the Bitcoin OTC dataset to simulate real-time graph evolution.
   - Mapped trust/distrust scores to binary labels (Trusted `0` vs. Fraudulent `1`).
   - Constructed rich historical features (e.g., node interaction counts, historical negative rating ratios) computed *strictly* from past interactions prior to each event to ensure **zero data leakage**.
2. **Temporal Graph Construction:** 
   - Constructed a dynamic graph where nodes and edges stream in over time, updating the memory footprint dynamically.
3. **Training & Contrastive Learning:**
   - **Imbalanced Learning:** Addressed the severe class imbalance (~10% fraud rate) by applying cost-sensitive weighted Binary Cross Entropy Loss (`BCEWithLogitsLoss`).
   - **Temporal Negative Sampling:** For every true interaction, sampled a random "fake" target interaction at the exact same timestamp to teach the model to discern true future edges from structural noise.
4. **Evaluation:**
   - Monitored model PR-AUC and ROC-AUC over time using Learning Rate (`ReduceLROnPlateau`) scheduling. 
   - Evaluated using strict classification metrics on chronological splits, focusing specifically on capturing the minority fraud class (High Recall scenario).

## 🏆 Final Results

The Temporal GNN achieved strong performance in identifying fraudulent behavior on the network.

### Overall Performance

| Model | Dataset | ROC-AUC | PR-AUC | Fraud Rate | Framework |
|-------|---------|---------|--------|------------|-----------|
| Temporal GNN (GRU + Memory) | Bitcoin OTC Trust Network | **0.8857** | **0.7131** | 10% | PyTorch |

### Classification Report (Threshold = 0.5)

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| **Trusted (0)** | 0.95 | 0.84 | 0.89 | 6024 |
| **Fraudulent (1)** | 0.47 | 0.77 | 0.58 | 1095 |
| *Accuracy* | | | **0.83** | *7119* |
| *Macro Avg* | 0.71 | 0.81 | 0.74 | *7119* |
| *Weighted Avg* | 0.88 | 0.83 | 0.85 | *7119* |

> *Note: The model achieves a high recall (0.77) for the Fraudulent class. In the context of risk mitigation and fraud detection, missing a fraudulent transaction (false negative) is typically far costlier than flagging a legitimate one for review (false positive).*

## 🚀 Getting Started

### Prerequisites

Ensure you have the following installed:
- Python 3.8+
- PyTorch
- Jupyter Notebook
- `pandas`, `numpy`, `scikit-learn`, `matplotlib`

### Running the Project

1. Clone this repository.
2. Ensure the dataset is placed in the `Dataset/` directory: `Dataset/soc-sign-bitcoinotc.csv`.
3. Open `TGNN.ipynb` in Jupyter Notebook or VS Code.
4. Run the cells sequentially to preprocess the data, train the Temporal GNN, and reproduce the results.
