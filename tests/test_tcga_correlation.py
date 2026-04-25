from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from nsclc_tf_switch.tcga_validation import compute_anchor_edge_tcga_validation


def _synthetic_cohort(tmp_path: Path, cohort: str, n_samples: int = 200) -> Path:
    rng = np.random.default_rng(0 if cohort == "LUAD" else 1)
    samples = [f"TCGA-{cohort}-{i:04d}" for i in range(n_samples)]
    exhaustion_latent = rng.normal(0, 1, size=n_samples)
    background = rng.normal(0, 1, size=(20, n_samples))
    # Exhaustion-signature genes track the latent exhaustion variable.
    exhaust_block = np.vstack(
        [exhaustion_latent + rng.normal(0, 0.2, size=n_samples) for _ in range(6)]
    )
    # CD274 + PDCD1 also track the exhaustion variable (target: strong correlation).
    anchor_block = np.vstack(
        [exhaustion_latent * 0.95 + rng.normal(0, 0.15, size=n_samples) for _ in range(2)]
    )
    # LGALS9 + HAVCR2 track exhaustion with slightly weaker coupling.
    lgals_block = np.vstack(
        [exhaustion_latent * 0.70 + rng.normal(0, 0.25, size=n_samples) for _ in range(2)]
    )
    # CD80 / CTLA4 / CD86 all moderate couplers.
    cd80_block = np.vstack(
        [exhaustion_latent * 0.60 + rng.normal(0, 0.3, size=n_samples) for _ in range(3)]
    )

    # Filler rows to hit >=20 genes total.
    filler = ["FILL_%d" % i for i in range(20)]
    matrix = np.vstack([
        exhaust_block,             # PDCD1..TOX
        anchor_block,              # CD274, PDCD1_alt
        lgals_block,               # LGALS9, HAVCR2_alt
        cd80_block,                # CD80, CTLA4_alt, CD86
        background,                # filler
    ])
    index = [
        "PDCD1", "CTLA4", "LAG3", "TIGIT", "HAVCR2", "TOX",
        "CD274", "PDCD1",   # duplicate to represent the same gene symbol
        "LGALS9", "HAVCR2",
        "CD80", "CTLA4", "CD86",
        *filler,
    ]
    # Consolidate duplicate symbols by max (mirrors Xena convention).
    df = pd.DataFrame(matrix, index=index, columns=samples)
    df = df.groupby(level=0).max()
    parquet = tmp_path / f"TCGA-{cohort}.symbol_fpkm.parquet"
    df.astype(np.float32).to_parquet(parquet)
    return parquet


def test_correlation_recovers_injected_signal(tmp_path: Path) -> None:
    paths = {
        "LUAD": _synthetic_cohort(tmp_path, "LUAD"),
        "LUSC": _synthetic_cohort(tmp_path, "LUSC"),
    }
    out = compute_anchor_edge_tcga_validation(
        paths,
        output_csv=tmp_path / "tcga_anchor_edge_correlations.csv",
        scatter_output_csv=tmp_path / "scatter.csv",
    )
    df = pd.read_csv(out)
    assert (df["n_samples"] >= 200).all()
    # All anchor edges are simulated as strongly positive; BH q must be < 0.1.
    for edge in (
        "b_plasma_CD274_PDCD1",
        "cancer_CD274_PDCD1",
        "treg_LGALS9_HAVCR2",
        "b_plasma_CD80_CTLA4",
    ):
        hits = df[df["edge"] == edge]
        rhos = hits["spearman_rho"].tolist()
        assert (hits["spearman_rho"] > 0.5).all(), f"{edge} rho too low: {rhos}"
        assert (hits["spearman_qvalue"] < 0.1).all(), f"{edge} q too high"


def test_bootstrap_ci_brackets_point_estimate(tmp_path: Path) -> None:
    paths = {"LUAD": _synthetic_cohort(tmp_path, "LUAD", n_samples=150)}
    out = compute_anchor_edge_tcga_validation(
        paths,
        output_csv=tmp_path / "corr.csv",
        scatter_output_csv=None,
    )
    df = pd.read_csv(out)
    finite = df.dropna(subset=["spearman_rho", "rho_ci_low", "rho_ci_high"])
    assert (finite["rho_ci_low"] <= finite["spearman_rho"]).all()
    assert (finite["spearman_rho"] <= finite["rho_ci_high"]).all()
