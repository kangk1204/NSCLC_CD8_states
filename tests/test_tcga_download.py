from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nsclc_tf_switch.tcga_validation import (
    EXHAUSTION_SIGNATURE_GENES,
    _expression_to_symbol_matrix,
    download_tcga_expression,
)


def test_download_tcga_expression_offline_short_circuit(tmp_path: Path) -> None:
    """With parquet files already present, download_tcga_expression must not hit the network."""
    luad_parquet = tmp_path / "TCGA-LUAD.symbol_fpkm.parquet"
    lusc_parquet = tmp_path / "TCGA-LUSC.symbol_fpkm.parquet"
    rng = np.random.default_rng(0)
    samples = [f"TCGA-LUAD-{i:03d}" for i in range(5)]
    df = pd.DataFrame(
        rng.normal(0, 1, size=(8, 5)).astype(np.float32),
        index=["PDCD1", "CTLA4", "LAG3", "TIGIT", "HAVCR2", "TOX", "CD274", "LGALS9"],
        columns=samples,
    )
    df.to_parquet(luad_parquet)
    df.to_parquet(lusc_parquet)

    # Should be a no-op because parquets already exist.
    paths = download_tcga_expression(output_dir=tmp_path)
    assert paths["LUAD"] == luad_parquet
    assert paths["LUSC"] == lusc_parquet


def test_expression_to_symbol_matrix_reads_gz(tmp_path: Path) -> None:
    import gzip

    source = tmp_path / "mini.tsv.gz"
    with gzip.open(source, "wt") as handle:
        handle.write("sample\tS1\tS2\n")
        handle.write("PDCD1\t2.5\t3.1\n")
        handle.write("CTLA4\t1.0\t1.5\n")
    matrix = _expression_to_symbol_matrix(source)
    assert matrix.shape == (2, 2)
    assert "PDCD1" in matrix.index
    assert matrix.index.name == "gene_symbol"


def test_cached_parquets_contain_key_genes() -> None:
    """If the real TCGA parquets exist locally, validate basic shape + gene presence."""
    parquet_dir = Path("data/raw/tcga")
    for cohort, min_samples in (("LUAD", 400), ("LUSC", 300)):
        path = parquet_dir / f"TCGA-{cohort}.symbol_fpkm.parquet"
        if not path.exists():
            pytest.skip(f"TCGA {cohort} parquet not cached locally")
        df = pd.read_parquet(path)
        assert df.shape[0] >= 15_000, f"{cohort} gene count too small"
        assert df.shape[1] >= min_samples, f"{cohort} sample count too small"
        for gene in (*EXHAUSTION_SIGNATURE_GENES, "CD274", "LGALS9", "CD80", "CD86"):
            assert gene in df.index, f"{cohort} missing gene {gene}"
