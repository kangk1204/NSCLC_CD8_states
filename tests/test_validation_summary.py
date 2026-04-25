from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nsclc_tf_switch.validation import build_validation_consensus


def _write_study_ranking(
    root: Path, study: str, rhos: dict[str, float], deltas: dict[str, float] | None = None
) -> None:
    deltas = deltas or rhos
    study_dir = root / study
    study_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "tf": tf,
                "spearman_rho": rho,
                "delta_exhausted_vs_activated": deltas.get(tf, rho),
                "switch_score": abs(rho),
            }
            for tf, rho in rhos.items()
        ]
    ).to_csv(study_dir / "tf_switch_ranking.csv", index=False)


@pytest.fixture()
def discovery_path(tmp_path: Path) -> Path:
    discovery = pd.DataFrame(
        [
            {
                "tf": "STAT2",
                "spearman_rho": 0.3,
                "delta_exhausted_vs_activated": 1.2,
                "switch_score": 0.7,
            },
            {
                "tf": "FOXO1",
                "spearman_rho": -0.2,
                "delta_exhausted_vs_activated": -0.8,
                "switch_score": 0.3,
            },
            {
                "tf": "MIXED",
                "spearman_rho": 0.4,
                "delta_exhausted_vs_activated": 0.4,
                "switch_score": 0.16,
            },
        ]
    )
    path = tmp_path / "discovery.csv"
    discovery.to_csv(path, index=False)
    return path


def test_build_validation_consensus(tmp_path: Path, discovery_path: Path) -> None:
    _write_study_ranking(
        tmp_path,
        "GSE131907",
        {"STAT2": 0.2, "FOXO1": -0.1, "MIXED": 0.15},
    )
    _write_study_ranking(
        tmp_path,
        "GSE136246",
        {"STAT2": 0.25, "FOXO1": -0.05, "MIXED": -0.2},
    )

    score_df, summary_df = build_validation_consensus(
        discovery_path,
        tmp_path,
        studies=("GSE131907", "GSE136246"),
    )
    assert "discovery_rho" in score_df.columns
    assert "GSE131907_rho" in score_df.columns
    assert "GSE136246_rho" in score_df.columns

    stat2 = summary_df[summary_df["tf"] == "STAT2"].iloc[0]
    assert stat2["validated_in_all_datasets"]
    assert stat2["direction_consensus"] == "positive"
    # CI columns exist and bracket the point estimate (external rhos only).
    for column in ("mean_abs_rho", "rho_ci_low", "rho_ci_high"):
        assert column in summary_df.columns
    expected_mean = float(np.mean([abs(0.2), abs(0.25)]))
    assert np.isclose(stat2["mean_abs_rho"], expected_mean, atol=1e-6)
    assert stat2["rho_ci_low"] <= stat2["mean_abs_rho"] <= stat2["rho_ci_high"]

    # MIXED has inconsistent direction; must not be marked validated.
    mixed = summary_df[summary_df["tf"] == "MIXED"].iloc[0]
    assert not mixed["validated_in_all_datasets"]
    assert mixed["direction_consensus"] in {"positive", "negative", "mixed"}


def test_validation_rule_is_internally_consistent(
    tmp_path: Path, discovery_path: Path
) -> None:
    """A TF missing from one cohort must not be marked validated."""
    _write_study_ranking(
        tmp_path, "GSE131907", {"STAT2": 0.2, "FOXO1": -0.1}  # STAT2 present
    )
    # GSE136246 missing STAT2 entirely.
    _write_study_ranking(tmp_path, "GSE136246", {"FOXO1": -0.05})

    _, summary_df = build_validation_consensus(
        discovery_path,
        tmp_path,
        studies=("GSE131907", "GSE136246"),
    )
    stat2 = summary_df[summary_df["tf"] == "STAT2"].iloc[0]
    assert stat2["finite_datasets"] == 2  # discovery + GSE131907
    assert not stat2["validated_in_all_datasets"]


def test_mean_abs_rho_nan_when_no_external(tmp_path: Path, discovery_path: Path) -> None:
    """When no external study records a rho, CI should be NaN rather than raise."""
    _, summary_df = build_validation_consensus(
        discovery_path,
        tmp_path,
        studies=(),
    )
    row = summary_df[summary_df["tf"] == "STAT2"].iloc[0]
    assert np.isnan(row["rho_ci_low"])
    assert np.isnan(row["rho_ci_high"])
