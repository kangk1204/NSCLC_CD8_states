from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.decomposition import TruncatedSVD
from sklearn.utils.sparsefuncs import mean_variance_axis

from nsclc_tf_switch.config import ACTIVATION_MARKERS, EXHAUSTION_MARKERS


def add_qc_metrics(adata: AnnData) -> None:
    counts = adata.layers.get("counts", adata.X)
    cell_counts = np.asarray(counts.sum(axis=1)).ravel()
    detected_genes = np.asarray((counts > 0).sum(axis=1)).ravel()
    mito_mask = np.array([name.upper().startswith("MT-") for name in adata.var_names], dtype=bool)
    mito_counts = (
        np.asarray(counts[:, mito_mask].sum(axis=1)).ravel()
        if mito_mask.any()
        else np.zeros(adata.n_obs)
    )
    adata.obs["total_counts"] = cell_counts
    adata.obs["n_genes_by_counts"] = detected_genes
    adata.obs["pct_counts_mt"] = np.divide(
        mito_counts,
        np.maximum(cell_counts, 1.0),
        out=np.zeros_like(mito_counts, dtype=float),
    )


def filter_basic_qc(
    adata: AnnData,
    min_counts: int = 500,
    min_genes: int = 200,
    max_mt_fraction: float = 0.2,
    min_cells_per_gene: int = 10,
) -> AnnData:
    add_qc_metrics(adata)
    keep_cells = (
        (adata.obs["total_counts"] >= min_counts)
        & (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["pct_counts_mt"] <= max_mt_fraction)
    )
    filtered = adata[keep_cells.values].copy()
    counts = filtered.layers.get("counts", filtered.X)
    keep_genes = np.asarray((counts > 0).sum(axis=0)).ravel() >= min_cells_per_gene
    return filtered[:, keep_genes].copy()


def normalize_log1p(adata: AnnData, target_sum: float = 1e4) -> None:
    counts = adata.layers.get("counts", adata.X).tocsr().astype(np.float32)
    totals = np.asarray(counts.sum(axis=1)).ravel()
    scale = target_sum / np.maximum(totals, 1.0)
    normalized = counts.multiply(scale[:, None]).tocsr()
    normalized.data = np.log1p(normalized.data)
    adata.X = normalized


def select_hvgs(adata: AnnData, n_top_genes: int = 1500) -> np.ndarray:
    means, variances = mean_variance_axis(adata.X, axis=0)
    dispersion = variances / np.maximum(means, 1e-6)
    order = np.argsort(dispersion)[::-1]
    n_keep = min(n_top_genes, len(order))
    mask = np.zeros(adata.n_vars, dtype=bool)
    mask[order[:n_keep]] = True
    adata.var["highly_variable"] = mask
    adata.var["mean"] = means
    adata.var["variance"] = variances
    adata.var["dispersion"] = dispersion
    return mask


def compute_svd_features(adata: AnnData, n_components: int = 32) -> np.ndarray:
    mask = adata.var.get("highly_variable", pd.Series(True, index=adata.var_names)).to_numpy()
    X = adata.X[:, mask]
    n_components = min(n_components, max(2, min(X.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=0)
    features = svd.fit_transform(X)
    adata.obsm["X_svd"] = features
    return features


def _score_marker_set(adata: AnnData, markers: list[str]) -> np.ndarray:
    markers_upper = {gene.upper() for gene in markers}
    mask = np.array([name.upper() in markers_upper for name in adata.var_names], dtype=bool)
    if not mask.any():
        return np.zeros(adata.n_obs, dtype=float)
    subset = adata.X[:, mask]
    scores = np.asarray(subset.mean(axis=1)).ravel()
    return scores


def add_transition_marker_scores(adata: AnnData) -> None:
    adata.obs["activation_score"] = _score_marker_set(adata, ACTIVATION_MARKERS)
    adata.obs["exhaustion_score"] = _score_marker_set(adata, EXHAUSTION_MARKERS)
    adata.obs["marker_delta"] = adata.obs["exhaustion_score"] - adata.obs["activation_score"]
