from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
from scipy import sparse

from nsclc_tf_switch.loom_io import load_loom_as_anndata, summarize_loom


def create_test_loom(path: Path) -> None:
    matrix = np.array(
        [
            [1, 0, 3, 0],
            [0, 5, 0, 1],
            [2, 2, 0, 0],
        ],
        dtype=np.int32,
    )
    with h5py.File(path, "w") as handle:
        handle.create_dataset("matrix", data=matrix)
        row_attrs = handle.create_group("row_attrs")
        row_attrs.create_dataset("Gene", data=np.array([b"CD3D", b"PDCD1", b"TOX"]))
        col_attrs = handle.create_group("col_attrs")
        col_attrs.create_dataset("CellID", data=np.array([b"c1", b"c2", b"c3", b"c4"]))
        col_attrs.create_dataset("Patient", data=np.array([b"p1", b"p1", b"p2", b"p2"]))


def test_summarize_loom(tmp_path: Path) -> None:
    loom_path = tmp_path / "toy.loom"
    create_test_loom(loom_path)
    summary = summarize_loom(loom_path)
    assert summary["shape"] == (3, 4)
    assert "Gene" in summary["row_attrs"]
    assert "CellID" in summary["col_attrs"]


def test_load_loom_as_anndata(tmp_path: Path) -> None:
    loom_path = tmp_path / "toy.loom"
    create_test_loom(loom_path)
    adata = load_loom_as_anndata(loom_path)
    assert adata.shape == (4, 3)
    assert sparse.isspmatrix_csr(adata.X)
    assert adata.obs_names.tolist() == ["c1", "c2", "c3", "c4"]
    assert adata.var_names.tolist() == ["CD3D", "PDCD1", "TOX"]
