from __future__ import annotations

from pathlib import Path

import pandas as pd

from nsclc_tf_switch.interaction_validation import build_interaction_validation_consensus


def test_build_interaction_validation_consensus(tmp_path: Path) -> None:
    discovery = pd.DataFrame(
        [
            {
                "sender_group": "myeloid",
                "ligand": "LGALS9",
                "receptor": "HAVCR2",
                "pathway": "checkpoint",
                "delta_score": 1.0,
                "association_rho": 0.5,
                "association_pvalue": 0.08,
            }
        ]
    )
    discovery_path = tmp_path / "discovery.csv"
    discovery.to_csv(discovery_path, index=False)

    for study, delta in {"GSE131907": 0.8, "GSE127465": 0.6}.items():
        study_dir = tmp_path / study
        study_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "sender_group": "myeloid",
                    "ligand": "LGALS9",
                    "receptor": "HAVCR2",
                    "pathway": "checkpoint",
                    "delta_score": delta,
                    "association_rho": 0.4,
                    "association_pvalue": 0.04,
                }
            ]
        ).to_csv(study_dir / "interaction_network.csv", index=False)

    scores, summary = build_interaction_validation_consensus(discovery_path, tmp_path)
    assert "discovery_delta" in scores.columns
    assert "GSE131907_delta" in scores.columns
    row = summary[
        (summary["sender_group"] == "myeloid")
        & (summary["ligand"] == "LGALS9")
        & (summary["receptor"] == "HAVCR2")
    ].iloc[0]
    assert row["validated_in_all_datasets"]


def test_interaction_validation_strict_rule_requires_two_external_q_hits(tmp_path: Path) -> None:
    discovery = pd.DataFrame(
        [
            {
                "sender_group": "myeloid",
                "ligand": "LGALS9",
                "receptor": "HAVCR2",
                "pathway": "checkpoint",
                "delta_score": 1.0,
                "association_rho": 0.5,
                "association_pvalue": 0.04,
            }
        ]
    )
    discovery_path = tmp_path / "discovery.csv"
    discovery.to_csv(discovery_path, index=False)

    study_payloads = {
        "GSE131907": {"delta_score": 0.8, "association_rho": 0.4, "association_pvalue": 0.03},
        "GSE127465": {"delta_score": 0.7, "association_rho": 0.3, "association_pvalue": 0.3},
    }
    for study, payload in study_payloads.items():
        study_dir = tmp_path / study
        study_dir.mkdir()
        pd.DataFrame(
            [
                {
                    "sender_group": "myeloid",
                    "ligand": "LGALS9",
                    "receptor": "HAVCR2",
                    "pathway": "checkpoint",
                    **payload,
                }
            ]
        ).to_csv(study_dir / "interaction_network.csv", index=False)

    _, summary = build_interaction_validation_consensus(
        discovery_path,
        tmp_path,
        studies=("GSE131907", "GSE127465"),
    )
    row = summary[
        (summary["sender_group"] == "myeloid")
        & (summary["ligand"] == "LGALS9")
        & (summary["receptor"] == "HAVCR2")
    ].iloc[0]
    assert row["discovery_significant"]
    assert row["significant_external_datasets"] == 1
    assert not row["validated_in_all_datasets"]
