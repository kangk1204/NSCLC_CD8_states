from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from anndata import AnnData
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch_geometric.nn import GAE, GCNConv
from torch_geometric.utils import from_scipy_sparse_matrix


class GCNEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, latent_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return x


def build_knn_graph(features: np.ndarray, n_neighbors: int = 15):
    model = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="cosine")
    model.fit(features)
    adjacency = model.kneighbors_graph(features, mode="connectivity")
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    edge_index, _ = from_scipy_sparse_matrix(adjacency.tocoo())
    return adjacency, edge_index


def fit_graph_autoencoder(
    features: np.ndarray,
    edge_index: torch.Tensor,
    epochs: int = 40,
    hidden_channels: int = 32,
    latent_dim: int = 2,
    learning_rate: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    torch.manual_seed(seed)
    x = torch.tensor(features, dtype=torch.float32)
    model = GAE(GCNEncoder(x.shape[1], hidden_channels, latent_dim))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        z = model.encode(x, edge_index)
        loss = model.recon_loss(z, edge_index)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        latent = model.encode(x, edge_index).cpu().numpy()
    return latent


def infer_transition_states(
    adata: AnnData,
    low_quantile: float = 0.15,
    high_quantile: float = 0.85,
) -> pd.DataFrame:
    delta = adata.obs["marker_delta"].to_numpy()
    low_cut = np.quantile(delta, low_quantile)
    high_cut = np.quantile(delta, high_quantile)

    labels = np.full(adata.n_obs, -1, dtype=int)
    labels[delta <= low_cut] = 0
    labels[delta >= high_cut] = 1

    train_mask = labels >= 0
    classifier = LogisticRegression(max_iter=1000, random_state=0)
    classifier.fit(adata.obsm["X_latent"][train_mask], labels[train_mask])
    probabilities = classifier.predict_proba(adata.obsm["X_latent"])[:, 1]

    transition_state = np.where(
        probabilities >= 0.8,
        "exhausted_anchor",
        np.where(probabilities <= 0.2, "activated_anchor", "transition_boundary"),
    )
    if not np.any(transition_state == "activated_anchor") or not np.any(
        transition_state == "exhausted_anchor"
    ):
        probabilities = (delta - delta.min()) / max(delta.max() - delta.min(), 1e-6)
        transition_state = np.where(
            delta >= high_cut,
            "exhausted_anchor",
            np.where(delta <= low_cut, "activated_anchor", "transition_boundary"),
        )
    adata.obs["transition_probability"] = probabilities
    adata.obs["transition_state"] = pd.Categorical(
        transition_state,
        categories=["activated_anchor", "transition_boundary", "exhausted_anchor"],
        ordered=True,
    )
    adata.obs["anchor_label"] = labels
    return adata.obs[["transition_probability", "transition_state", "anchor_label"]].copy()
