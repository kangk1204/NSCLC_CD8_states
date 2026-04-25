from __future__ import annotations

import numpy as np
from anndata import AnnData
from scipy import sparse

from nsclc_tf_switch.model import (
    build_knn_graph,
    fit_graph_autoencoder,
    infer_transition_states,
)
from nsclc_tf_switch.preprocess import (
    add_transition_marker_scores,
    compute_svd_features,
    select_hvgs,
)
from nsclc_tf_switch.tf_activity import rank_tf_switches


def test_graph_transition_and_tf_ranking_smoke() -> None:
    rng = np.random.default_rng(0)
    genes = ["IL7R", "LTB", "GZMK", "PDCD1", "CTLA4", "LAG3", "TOX", "TIGIT", "ACTB", "MALAT1"]
    cells = 60
    counts = rng.poisson(1.0, size=(cells, len(genes))).astype(np.float32)
    counts[:20, :3] += 4
    counts[40:, 3:8] += 5
    adata = AnnData(X=sparse.csr_matrix(counts))
    adata.layers["counts"] = adata.X.copy()
    adata.var_names = genes

    select_hvgs(adata, n_top_genes=10)
    compute_svd_features(adata, n_components=4)
    add_transition_marker_scores(adata)
    _, edge_index = build_knn_graph(adata.obsm["X_svd"], n_neighbors=8)
    adata.obsm["X_latent"] = fit_graph_autoencoder(
        adata.obsm["X_svd"],
        edge_index,
        epochs=5,
        hidden_channels=8,
    )
    infer_transition_states(adata)

    adata.obsm["tf_activity"] = adata.to_df()[["TOX", "PDCD1", "LTB"]].rename(
        columns={"TOX": "TOX", "PDCD1": "NR4A1", "LTB": "TCF7"}
    )
    ranking = rank_tf_switches(adata, top_n=3)

    assert not ranking.empty
    assert ranking.iloc[0]["tf"] in {"TOX", "NR4A1", "TCF7"}
