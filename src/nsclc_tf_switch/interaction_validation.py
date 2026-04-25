from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from nsclc_tf_switch.config import DEFAULT_INTERACTION_VALIDATION_STUDIES, INTERACTION_PANEL
from nsclc_tf_switch.interaction import analyze_matrix_market_interactions
from nsclc_tf_switch.reporting import save_interaction_validation_heatmap
from nsclc_tf_switch.validation import materialize_validation_studies


def _benjamini_hochberg(pvalues: pd.Series) -> pd.Series:
    values = pvalues.astype(float).copy()
    mask = values.notna()
    if not mask.any():
        return pd.Series(index=values.index, dtype=float)
    ranked = values[mask].sort_values()
    m = len(ranked)
    adjusted = ranked * m / pd.Series(range(1, m + 1), index=ranked.index, dtype=float)
    adjusted = adjusted[::-1].cummin()[::-1].clip(upper=1.0)
    result = pd.Series(index=values.index, dtype=float)
    result.loc[adjusted.index] = adjusted
    return result


def build_interaction_validation_consensus(
    discovery_interaction_path: str | Path,
    validation_root: str | Path,
    studies: tuple[str, ...] = DEFAULT_INTERACTION_VALIDATION_STUDIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    discovery = pd.read_csv(discovery_interaction_path)
    edge_panel = discovery.head(20)[
        ["sender_group", "ligand", "receptor", "pathway"]
    ].drop_duplicates()
    curated = pd.DataFrame(
        [
            {
                "sender_group": sender_group,
                "ligand": pair.ligand,
                "receptor": pair.receptor,
                "pathway": pair.pathway,
            }
            for sender_group in [
                "myeloid",
                "cancer",
                "endothelial",
                "treg",
                "fibroblast",
                "b_plasma",
            ]
            for pair in INTERACTION_PANEL
            if pair.pathway == "checkpoint"
        ]
    )
    panel = pd.concat([edge_panel, curated], ignore_index=True).drop_duplicates()

    datasets = {"discovery": discovery}
    for study in studies:
        datasets[study] = pd.read_csv(Path(validation_root) / study / "interaction_network.csv")
    dataset_qvalues: dict[str, pd.Series] = {}
    for dataset_name, df in datasets.items():
        if "association_pvalue" in df.columns:
            dataset_qvalues[dataset_name] = _benjamini_hochberg(df["association_pvalue"])
        else:
            dataset_qvalues[dataset_name] = pd.Series(index=df.index, dtype=float)

    score_rows: list[dict[str, float | str | None]] = []
    summary_rows: list[dict[str, float | str | int | bool]] = []
    for edge in panel.to_dict(orient="records"):
        score_row: dict[str, float | str | None] = edge.copy()
        positive = 0
        negative = 0
        present = 0
        rho_present = 0
        positive_rho = 0
        discovery_significant = False
        significant_external = 0
        for dataset_name, df in datasets.items():
            hit = df[
                (df["sender_group"] == edge["sender_group"])
                & (df["ligand"] == edge["ligand"])
                & (df["receptor"] == edge["receptor"])
            ]
            delta = float(hit.iloc[0]["delta_score"]) if not hit.empty else None
            rho = (
                float(hit.iloc[0]["association_rho"])
                if not hit.empty and "association_rho" in hit.columns
                else None
            )
            pvalue = (
                float(hit.iloc[0]["association_pvalue"])
                if not hit.empty and "association_pvalue" in hit.columns
                else None
            )
            qvalue = (
                float(dataset_qvalues[dataset_name].loc[hit.index[0]])
                if not hit.empty
                and dataset_name in dataset_qvalues
                and hit.index[0] in dataset_qvalues[dataset_name].index
                and pd.notna(dataset_qvalues[dataset_name].loc[hit.index[0]])
                else None
            )
            score_row[f"{dataset_name}_delta"] = delta
            score_row[f"{dataset_name}_rho"] = rho
            score_row[f"{dataset_name}_pvalue"] = pvalue
            score_row[f"{dataset_name}_qvalue"] = qvalue
            if (
                delta is not None
                and pd.notna(delta)
                and rho is not None
                and pd.notna(rho)
            ):
                present += 1
                rho_present += 1
                if delta >= 0:
                    positive += 1
                else:
                    negative += 1
                if rho > 0:
                    positive_rho += 1
                if (
                    qvalue is not None
                    and pd.notna(qvalue)
                    and qvalue < 0.10
                    and delta > 0
                    and rho > 0
                ):
                    if dataset_name == "discovery":
                        discovery_significant = True
                    else:
                        significant_external += 1
        score_rows.append(score_row)
        summary_rows.append(
            {
                **edge,
                "datasets_present": present,
                "datasets_with_finite_rho": rho_present,
                "positive_delta_datasets": positive,
                "negative_delta_datasets": negative,
                "positive_rho_datasets": positive_rho,
                "discovery_significant": discovery_significant,
                "significant_external_datasets": significant_external,
                "direction_consensus": (
                    "positive"
                    if positive > negative
                    else "negative"
                    if negative > positive
                    else "mixed"
                ),
                "validated_in_all_datasets": (
                    present == len(datasets)
                    and positive == len(datasets)
                    and positive_rho == len(datasets)
                    and discovery_significant
                    and significant_external >= 2
                ),
            }
        )

    return pd.DataFrame(score_rows), pd.DataFrame(summary_rows)


def run_interaction_validation(
    discovery_interaction_path: str | Path,
    raw_matrix_rds: str | Path,
    metadata_csv: str | Path,
    output_dir: str | Path,
    studies: tuple[str, ...] = DEFAULT_INTERACTION_VALIDATION_STUDIES,
    epochs: int = 25,
) -> Path:
    output_path = Path(output_dir)
    materialized_dir = output_path / "materialized"
    materialize_validation_studies(
        raw_matrix_rds=raw_matrix_rds,
        metadata_csv=metadata_csv,
        output_dir=materialized_dir,
        studies=studies,
        cell_level1="ALL",
    )

    for study in studies:
        analyze_matrix_market_interactions(
            matrix_dir=materialized_dir / study,
            study=study,
            output_dir=output_path / study,
            epochs=epochs,
        )

    scores, summary = build_interaction_validation_consensus(
        discovery_interaction_path=discovery_interaction_path,
        validation_root=output_path,
        studies=studies,
    )
    scores.to_csv(output_path / "interaction_validation_scores.csv", index=False)
    summary.to_csv(output_path / "interaction_validation_summary.csv", index=False)
    save_interaction_validation_heatmap(scores, output_path)
    (output_path / "interaction_validation.json").write_text(
        json.dumps(
            {
                "studies": list(studies),
                "validated_edges": summary[summary["validated_in_all_datasets"]]
                .head(20)
                .to_dict(orient="records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path
