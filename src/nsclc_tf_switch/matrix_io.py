from __future__ import annotations

from pathlib import Path

import anndata as ad
import pandas as pd
from scipy import io, sparse


def load_matrix_market_as_anndata(
    matrix_path: str | Path,
    features_path: str | Path,
    obs_path: str | Path,
    source_dataset: str,
) -> ad.AnnData:
    matrix = io.mmread(matrix_path)
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    matrix = matrix.tocsr().T.tocsr()

    obs = pd.read_csv(obs_path)
    if "CellID" not in obs.columns:
        first_column = obs.columns[0]
        obs = obs.rename(columns={first_column: "CellID"})
    obs["CellID"] = obs["CellID"].astype(str)
    obs.index = obs["CellID"]

    features = pd.read_csv(features_path, sep="\t", header=None, names=["gene_symbol"])
    features["gene_symbol"] = features["gene_symbol"].astype(str)
    var = pd.DataFrame(index=pd.Index(features["gene_symbol"], name="gene_symbol"))

    adata = ad.AnnData(X=matrix, obs=obs, var=var)
    adata.var_names_make_unique()
    adata.layers["counts"] = adata.X.copy()
    adata.obs["source_dataset"] = source_dataset
    return adata
