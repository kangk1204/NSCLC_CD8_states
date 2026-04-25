from __future__ import annotations

from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

STRING_DTYPE_KINDS = {"S", "U", "O"}


def _decode_1d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind in STRING_DTYPE_KINDS:
        return np.array(
            [
                item.decode("utf-8") if isinstance(item, (bytes, bytearray)) else str(item)
                for item in array
            ],
            dtype=object,
        )
    return array


def _pick_first_key(group: h5py.Group, candidates: list[str], fallback_size: int) -> np.ndarray:
    for key in candidates:
        if key in group:
            return _decode_1d(group[key][...])
    return np.array([f"item_{idx}" for idx in range(fallback_size)], dtype=object)


def _is_supported_vector(dataset: h5py.Dataset, expected_size: int) -> bool:
    return (
        dataset.ndim == 1
        and dataset.shape[0] == expected_size
        and dataset.dtype.fields is None
        and dataset.dtype.kind != "V"
    )


def summarize_loom(path: str | Path) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        row_attrs = list(handle.get("row_attrs", {}).keys())
        col_attrs = list(handle.get("col_attrs", {}).keys())
        return {
            "path": str(path),
            "shape": tuple(int(v) for v in matrix.shape),
            "dtype": str(matrix.dtype),
            "row_attrs": row_attrs,
            "col_attrs": col_attrs,
        }


def _read_obs(group: h5py.Group, n_cells: int) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}
    for key, dataset in group.items():
        if not _is_supported_vector(dataset, n_cells):
            continue
        data[key] = _decode_1d(dataset[...])
    if "CellID" not in data:
        data["CellID"] = np.array([f"cell_{idx}" for idx in range(n_cells)], dtype=object)
    obs = pd.DataFrame(data)
    obs.index = obs["CellID"].astype(str)
    return obs


def _read_var(group: h5py.Group, n_genes: int) -> pd.DataFrame:
    gene_names = _pick_first_key(group, ["Gene", "GeneName", "GeneSymbol", "Symbol"], n_genes)
    var = pd.DataFrame(index=pd.Index(gene_names.astype(str), name="gene_symbol"))
    for key, dataset in group.items():
        if not _is_supported_vector(dataset, n_genes):
            continue
        var[key] = _decode_1d(dataset[...])
    return var


def load_loom_as_anndata(
    path: str | Path,
    chunk_size_cells: int = 256,
    max_cells: int | None = None,
) -> ad.AnnData:
    with h5py.File(path, "r") as handle:
        matrix = handle["matrix"]
        n_genes, n_cells = matrix.shape
        obs = _read_obs(handle["col_attrs"], n_cells)
        var = _read_var(handle["row_attrs"], n_genes)

        selected = np.arange(n_cells)
        if max_cells is not None and max_cells < n_cells:
            selected = selected[:max_cells]
            obs = obs.iloc[selected].copy()

        chunks: list[sparse.csr_matrix] = []
        for start in range(0, len(selected), chunk_size_cells):
            cell_idx = selected[start : start + chunk_size_cells]
            block = matrix[:, cell_idx]
            block_2d = np.asarray(block)
            if block_2d.ndim == 1:
                block_2d = block_2d[:, None]
            chunks.append(sparse.csr_matrix(block_2d.T))

        counts = sparse.vstack(chunks, format="csr")
        adata = ad.AnnData(X=counts, obs=obs, var=var)
        adata.layers["counts"] = counts.copy()
        adata.obs["source_dataset"] = "ku_loom"
        return adata
