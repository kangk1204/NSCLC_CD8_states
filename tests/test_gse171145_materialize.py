from __future__ import annotations

import gzip
import tarfile
from io import BytesIO
from pathlib import Path

import pandas as pd

from nsclc_tf_switch.validation import materialize_gse171145_tcells


def _add_tar_member(tar: tarfile.TarFile, name: str, content: str) -> None:
    data = gzip.compress(content.encode("utf-8"))
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, BytesIO(data))


def test_materialize_gse171145_tcells_handles_gene_order_mismatch(tmp_path: Path) -> None:
    tar_path = tmp_path / "GSE171145_RAW.tar"
    with tarfile.open(tar_path, "w") as tar:
        _add_tar_member(
            tar,
            "GSM1_A.counts.tsv.gz",
            "gene\tC1\tC2\nG1\t1\t0\nG2\t0\t2\n",
        )
        _add_tar_member(
            tar,
            "GSM1_A.cellname.list.txt.gz",
            "CellName\tCellIndex\ncellA\tC1\ncellB\tC2\n",
        )
        _add_tar_member(
            tar,
            "GSM2_B.counts.tsv.gz",
            "gene\tC1\nG2\t3\nG1\t4\n",
        )
        _add_tar_member(
            tar,
            "GSM2_B.cellname.list.txt.gz",
            "CellName\tCellIndex\ncellC\tC1\n",
        )

    out_dir = materialize_gse171145_tcells(tar_path, tmp_path / "out")
    assert (out_dir / "matrix.mtx").exists()
    assert (out_dir / "features.tsv").exists()
    assert (out_dir / "obs.csv").exists()

    features = pd.read_csv(out_dir / "features.tsv", sep="\t", header=None)
    obs = pd.read_csv(out_dir / "obs.csv")
    assert features[0].tolist() == ["G1", "G2"]
    assert obs["CellID"].tolist() == ["GSM1_A_cellA", "GSM1_A_cellB", "GSM2_B_cellC"]
