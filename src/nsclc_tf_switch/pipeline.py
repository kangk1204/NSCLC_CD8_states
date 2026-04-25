from __future__ import annotations

from pathlib import Path

from anndata import AnnData

from nsclc_tf_switch.loom_io import load_loom_as_anndata
from nsclc_tf_switch.model import build_knn_graph, fit_graph_autoencoder, infer_transition_states
from nsclc_tf_switch.preprocess import (
    add_transition_marker_scores,
    compute_svd_features,
    filter_basic_qc,
    normalize_log1p,
    select_hvgs,
)
from nsclc_tf_switch.reporting import (
    save_run_summary,
    save_top_tf_boxplot,
    save_transition_embedding,
)
from nsclc_tf_switch.tf_activity import rank_tf_switches, score_tf_activity


def prepare_anndata(adata: AnnData) -> AnnData:
    adata = filter_basic_qc(adata)
    normalize_log1p(adata)
    select_hvgs(adata)
    compute_svd_features(adata)
    add_transition_marker_scores(adata)
    return adata


def prepare_from_loom(path: str | Path, max_cells: int | None = None) -> AnnData:
    adata = load_loom_as_anndata(path, max_cells=max_cells)
    return prepare_anndata(adata)


def run_graph_tf_analysis(
    adata: AnnData,
    output_dir: str | Path,
    epochs: int = 40,
    hidden_channels: int = 32,
    latent_dim: int = 2,
    n_neighbors: int = 15,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prepared_path = output_path / "prepared.h5ad"
    adata.write_h5ad(prepared_path)

    _, edge_index = build_knn_graph(adata.obsm["X_svd"], n_neighbors=n_neighbors)
    adata.obsm["X_latent"] = fit_graph_autoencoder(
        adata.obsm["X_svd"],
        edge_index=edge_index,
        epochs=epochs,
        hidden_channels=hidden_channels,
        latent_dim=latent_dim,
    )
    infer_transition_states(adata)
    score_tf_activity(adata)
    ranking = rank_tf_switches(adata)

    modeled_path = output_path / "modeled.h5ad"
    ranking_path = output_path / "tf_switch_ranking.csv"

    adata.write_h5ad(modeled_path)
    ranking.to_csv(ranking_path, index=False)
    save_transition_embedding(adata, output_path)
    save_top_tf_boxplot(adata, ranking, output_path)
    save_run_summary(adata, ranking, output_path)
    return output_path


def analyze_loom(
    path: str | Path,
    output_dir: str | Path,
    max_cells: int | None = None,
    epochs: int = 40,
) -> Path:
    adata = prepare_from_loom(path, max_cells=max_cells)
    return run_graph_tf_analysis(adata, output_dir=output_dir, epochs=epochs)
