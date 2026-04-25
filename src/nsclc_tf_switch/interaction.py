from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import spearmanr

from nsclc_tf_switch.config import INTERACTION_PANEL
from nsclc_tf_switch.loom_io import load_loom_as_anndata
from nsclc_tf_switch.matrix_io import load_matrix_market_as_anndata
from nsclc_tf_switch.preprocess import (
    add_transition_marker_scores,
    filter_basic_qc,
    normalize_log1p,
)
from nsclc_tf_switch.reporting import save_interaction_heatmap, save_patient_association_plot


def _get_patient_column(obs: pd.DataFrame) -> str:
    for column in ["PatientNumber", "Patient", "Sample", "patient_id"]:
        if column in obs.columns:
            return column
    raise KeyError("Could not infer patient identifier column.")


def _get_cluster_columns(obs: pd.DataFrame) -> tuple[str | None, str | None]:
    level1 = "Cell_Cluster_level1" if "Cell_Cluster_level1" in obs.columns else None
    level2 = "Cell_Cluster_level2" if "Cell_Cluster_level2" in obs.columns else None
    return level1, level2


def _sender_group_from_row(row: pd.Series) -> str | None:
    level1, level2 = _get_cluster_columns(row.to_frame().T)
    if level1 is not None:
        level1_value = str(row.get(level1, "")).strip()
        level2_value = str(row.get(level2, "")).strip() if level2 is not None else ""
        if level1_value in {"Myeloid", "Mast"}:
            return "myeloid"
        if level1_value == "Cancer":
            return "cancer"
        if level1_value == "Endothelial":
            return "endothelial"
        if level1_value == "Fibroblasts":
            return "fibroblast"
        if level1_value in {"B", "Plasma"}:
            return "b_plasma"
        if level1_value == "T" and "Treg" in level2_value:
            return "treg"
        return None

    label = str(row.get("ClusterName", "")).lower().strip()
    if "macrophage" in label or "langerhans" in label or "granulocyte" in label:
        return "myeloid"
    if "regulatory t cells" in label:
        return "treg"
    if "cancer cells" in label:
        return "cancer"
    if "endothelial" in label:
        return "endothelial"
    if "fibroblast" in label:
        return "fibroblast"
    if "b cells" in label or "plasma" in label:
        return "b_plasma"
    return None


def _receiver_mask_from_row(row: pd.Series) -> bool:
    level1, level2 = _get_cluster_columns(row.to_frame().T)
    if level1 is not None:
        level1_value = str(row.get(level1, "")).strip()
        level2_value = str(row.get(level2, "")).strip() if level2 is not None else ""
        return level1_value == "T" and "CD8" in level2_value
    label = str(row.get("ClusterName", "")).lower()
    return "cd8+ t cells" in label


def _genes_in_interaction_panel() -> tuple[list[str], list[str]]:
    ligands = sorted({pair.ligand for pair in INTERACTION_PANEL})
    receptors = sorted({pair.receptor for pair in INTERACTION_PANEL})
    return ligands, receptors


def _build_receiver_scores(adata: AnnData) -> pd.DataFrame:
    receiver = adata[adata.obs["is_receiver_cd8"].values].copy()
    add_transition_marker_scores(receiver)
    return receiver.obs[["activation_score", "exhaustion_score", "marker_delta"]].copy()


def _prepare_expression_frame(adata: AnnData, genes: list[str]) -> pd.DataFrame:
    present = [gene for gene in genes if gene in adata.var_names]
    expr = adata[:, present].to_df()
    expr.columns = [str(col) for col in expr.columns]
    return expr


def _group_mean(
    expr: pd.DataFrame,
    obs: pd.DataFrame,
    group_cols: list[str],
    min_cells: int,
) -> pd.DataFrame:
    cell_counts = obs.groupby(group_cols).size().rename("cell_count").reset_index()
    valid = cell_counts[cell_counts["cell_count"] >= min_cells]
    if valid.empty:
        return pd.DataFrame(columns=[*group_cols, *expr.columns.tolist(), "cell_count"])
    merged = expr.join(obs[group_cols])
    grouped = merged.groupby(group_cols, observed=True).mean().reset_index()
    return grouped.merge(valid, on=group_cols, how="inner")


def analyze_allcell_interactions(
    loom_path: str | Path,
    output_dir: str | Path,
    epochs: int = 25,
    min_sender_cells: int = 25,
    min_receiver_cells: int = 10,
) -> Path:
    adata = load_loom_as_anndata(loom_path)
    return analyze_allcell_interactions_adata(
        adata=adata,
        output_dir=output_dir,
        epochs=epochs,
        min_sender_cells=min_sender_cells,
        min_receiver_cells=min_receiver_cells,
    )


def analyze_matrix_market_interactions(
    matrix_dir: str | Path,
    study: str,
    output_dir: str | Path,
    epochs: int = 25,
    min_sender_cells: int = 25,
    min_receiver_cells: int = 10,
) -> Path:
    matrix_path = Path(matrix_dir)
    adata = load_matrix_market_as_anndata(
        matrix_path=matrix_path / "matrix.mtx",
        features_path=matrix_path / "features.tsv",
        obs_path=matrix_path / "obs.csv",
        source_dataset=study,
    )
    return analyze_allcell_interactions_adata(
        adata=adata,
        output_dir=output_dir,
        epochs=epochs,
        min_sender_cells=min_sender_cells,
        min_receiver_cells=min_receiver_cells,
    )


def analyze_allcell_interactions_adata(
    adata: AnnData,
    output_dir: str | Path,
    epochs: int = 25,
    min_sender_cells: int = 25,
    min_receiver_cells: int = 10,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    adata = filter_basic_qc(adata)
    normalize_log1p(adata)
    patient_column = _get_patient_column(adata.obs)
    adata.obs["patient_id"] = adata.obs[patient_column].astype(str)
    adata.obs["sender_group"] = adata.obs.apply(_sender_group_from_row, axis=1)
    adata.obs["is_receiver_cd8"] = adata.obs.apply(_receiver_mask_from_row, axis=1)
    receiver_scores = _build_receiver_scores(adata)
    adata.obs["receiver_marker_delta"] = np.nan
    adata.obs["receiver_activation_score"] = np.nan
    adata.obs["receiver_exhaustion_score"] = np.nan
    adata.obs.loc[receiver_scores.index, "receiver_marker_delta"] = receiver_scores["marker_delta"]
    adata.obs.loc[receiver_scores.index, "receiver_activation_score"] = (
        receiver_scores["activation_score"]
    )
    adata.obs.loc[receiver_scores.index, "receiver_exhaustion_score"] = (
        receiver_scores["exhaustion_score"]
    )

    ligands, receptors = _genes_in_interaction_panel()
    expr = _prepare_expression_frame(adata, genes=[*ligands, *receptors])

    sender_obs = adata.obs[adata.obs["sender_group"].notna()][["patient_id", "sender_group"]]
    receiver_obs = adata.obs[adata.obs["is_receiver_cd8"]][["patient_id"]]

    sender_means = _group_mean(
        expr.loc[sender_obs.index],
        sender_obs,
        ["patient_id", "sender_group"],
        min_sender_cells,
    )
    receiver_means = _group_mean(
        expr.loc[receiver_obs.index],
        receiver_obs,
        ["patient_id"],
        min_receiver_cells,
    )
    receiver_state_means = (
        adata.obs.loc[receiver_obs.index, ["patient_id", "receiver_marker_delta"]]
        .dropna()
        .groupby("patient_id", observed=True)["receiver_marker_delta"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "receiver_marker_delta_mean", "count": "receiver_cd8_cells"})
        .reset_index()
    )
    receiver_state_means = receiver_state_means[
        receiver_state_means["receiver_cd8_cells"] >= min_receiver_cells
    ]

    sender_means = sender_means.set_index(["patient_id", "sender_group"])
    receiver_means = receiver_means.set_index(["patient_id"])
    receiver_state_means = receiver_state_means.set_index("patient_id")

    interaction_columns = [
        "sender_group",
        "ligand",
        "receptor",
        "pathway",
        "n_patients",
        "mean_edge_score",
        "mean_receiver_marker_delta",
        "delta_score",
        "association_rho",
        "association_pvalue",
    ]

    rows: list[dict[str, float | str | int | None]] = []
    for sender_group in sorted(sender_obs["sender_group"].dropna().unique()):
        for pair in INTERACTION_PANEL:
            if (
                pair.ligand not in sender_means.columns
                or pair.receptor not in receiver_means.columns
            ):
                continue

            edge_scores: list[float] = []
            exhaustion_means: list[float] = []
            for patient_id in sorted(adata.obs["patient_id"].unique()):
                sender_key = (patient_id, sender_group)
                if sender_key not in sender_means.index:
                    continue
                if (
                    patient_id not in receiver_means.index
                    or patient_id not in receiver_state_means.index
                ):
                    continue
                ligand_value = float(sender_means.loc[sender_key, pair.ligand])
                edge_scores.append(
                    ligand_value * float(receiver_means.loc[patient_id, pair.receptor])
                )
                exhaustion_means.append(
                    float(receiver_state_means.loc[patient_id, "receiver_marker_delta_mean"])
                )

            if len(edge_scores) < 3:
                continue

            if np.allclose(np.std(edge_scores), 0.0) or np.allclose(np.std(exhaustion_means), 0.0):
                rho, pvalue = np.nan, np.nan
            else:
                rho, pvalue = spearmanr(edge_scores, exhaustion_means)
            edge_df = pd.DataFrame(
                {
                    "edge_score": edge_scores,
                    "receiver_marker_delta_mean": exhaustion_means,
                }
            )
            median_delta = edge_df["receiver_marker_delta_mean"].median()
            high_group = edge_df[edge_df["receiver_marker_delta_mean"] >= median_delta][
                "edge_score"
            ]
            low_group = edge_df[edge_df["receiver_marker_delta_mean"] < median_delta]["edge_score"]
            delta_score = (
                float(high_group.mean() - low_group.mean())
                if len(low_group) and len(high_group)
                else np.nan
            )

            rows.append(
                {
                    "sender_group": sender_group,
                    "ligand": pair.ligand,
                    "receptor": pair.receptor,
                    "pathway": pair.pathway,
                    "n_patients": len(edge_scores),
                    "mean_edge_score": float(np.mean(edge_scores)),
                    "mean_receiver_marker_delta": float(np.mean(exhaustion_means)),
                    "delta_score": delta_score,
                    "association_rho": float(rho) if pd.notna(rho) else np.nan,
                    "association_pvalue": float(pvalue) if pd.notna(pvalue) else np.nan,
                }
            )

    if rows:
        interaction_df = (
            pd.DataFrame(rows)
            .sort_values(["association_rho", "delta_score"], ascending=[False, False])
            .reset_index(drop=True)
        )
    else:
        interaction_df = pd.DataFrame(columns=interaction_columns)
    interaction_df.to_csv(output_path / "interaction_network.csv", index=False)

    patient_totals = adata.obs.groupby("patient_id").size()
    sender_counts = (
        adata.obs[adata.obs["sender_group"].notna()]
        .groupby(["patient_id", "sender_group"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    sender_fraction = sender_counts.div(patient_totals, axis=0).fillna(0.0)

    association_rows: list[dict[str, float | str | None]] = []
    for sender_group in sender_fraction.columns:
        common = receiver_state_means.index.intersection(sender_fraction.index)
        rho, pvalue = spearmanr(
            receiver_state_means.loc[common, "receiver_marker_delta_mean"].to_numpy(),
            sender_fraction.loc[common, sender_group].to_numpy(),
        )
        association_rows.append(
            {
                "sender_group": str(sender_group),
                "spearman_rho": float(rho),
                "pvalue": float(pvalue),
            }
        )

    if association_rows:
        association_df = pd.DataFrame(association_rows).sort_values("spearman_rho", ascending=False)
    else:
        association_df = pd.DataFrame(columns=["sender_group", "spearman_rho", "pvalue"])
    association_df.to_csv(output_path / "patient_associations.csv", index=False)

    save_interaction_heatmap(interaction_df, output_path)
    save_patient_association_plot(association_df, output_path)
    (output_path / "interaction_summary.json").write_text(
        json.dumps(
            {
                "top_edges": interaction_df.head(10).to_dict(orient="records"),
                "top_associations": association_df.head(10).to_dict(orient="records"),
                "receiver_exhaustion_metric": (
                    "patient-level mean CD8 marker_delta across all receiver CD8 cells"
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_path
