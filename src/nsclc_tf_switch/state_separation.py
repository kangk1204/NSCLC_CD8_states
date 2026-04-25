from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData, read_h5ad
from scipy.stats import mannwhitneyu

from nsclc_tf_switch.tf_activity import benjamini_hochberg

STATE_ORDER = ("activated_anchor", "transition_boundary", "exhausted_anchor")


def _tf_activity_frame(adata: AnnData) -> pd.DataFrame:
    tf_scores = adata.obsm["tf_activity"]
    if isinstance(tf_scores, pd.DataFrame):
        return tf_scores.copy()
    return pd.DataFrame(tf_scores, index=adata.obs_names)


def _ordered_states(state_values: pd.Series) -> list[str]:
    present = {str(value) for value in state_values.dropna().unique()}
    ordered = [state for state in STATE_ORDER if state in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def compute_tf_state_markers(
    adata: AnnData,
    min_group_size: int = 20,
) -> pd.DataFrame:
    """Run one-vs-rest Wilcoxon marker analysis on TF activity for each state.

    This mirrors the common single-cell marker workflow, but operates on TF
    activity scores rather than gene expression.
    """

    tf_scores = _tf_activity_frame(adata)
    states = adata.obs["transition_state"].astype(str)
    rows: list[dict[str, float | int | str | bool]] = []

    for state in _ordered_states(states):
        in_state = states == state
        n_target = int(in_state.sum())
        n_other = int((~in_state).sum())
        evaluable = n_target >= min_group_size and n_other >= min_group_size
        state_rows: list[dict[str, float | int | str | bool]] = []
        for tf in tf_scores.columns:
            values = tf_scores[tf].to_numpy(dtype=float)
            mean_target = float(np.nanmean(values[in_state])) if n_target else float("nan")
            mean_other = float(np.nanmean(values[~in_state])) if n_other else float("nan")
            delta_mean = float(mean_target - mean_other)
            direction = "high_in_state" if delta_mean >= 0 else "low_in_state"
            auroc = float("nan")
            pvalue = float("nan")
            if evaluable:
                test = mannwhitneyu(
                    values[in_state],
                    values[~in_state],
                    alternative="two-sided",
                    method="asymptotic",
                )
                auroc = float(test.statistic) / float(n_target * n_other)
                pvalue = float(test.pvalue)
            state_rows.append(
                {
                    "tf": str(tf),
                    "target_state": state,
                    "n_target": n_target,
                    "n_other": n_other,
                    "mean_target": mean_target,
                    "mean_other": mean_other,
                    "delta_mean": delta_mean,
                    "direction": direction,
                    "auroc": auroc,
                    "directional_auc": (
                        auroc if direction == "high_in_state" else 1.0 - auroc
                    )
                    if np.isfinite(auroc)
                    else float("nan"),
                    "separation_score": abs(auroc - 0.5) * 2.0
                    if np.isfinite(auroc)
                    else float("nan"),
                    "pvalue": pvalue,
                    "evaluable": evaluable,
                }
            )
        state_df = pd.DataFrame(state_rows)
        state_df["qvalue"] = benjamini_hochberg(state_df["pvalue"].to_numpy(dtype=float))
        rows.extend(state_df.to_dict(orient="records"))

    return pd.DataFrame(rows)


def summarize_top_state_tfs(
    marker_df: pd.DataFrame,
    top_n: int = 12,
) -> pd.DataFrame:
    evaluable = marker_df[marker_df["evaluable"]].copy()
    evaluable["direction_priority"] = (evaluable["direction"] == "high_in_state").astype(int)
    evaluable = evaluable.sort_values(
        ["direction_priority", "separation_score", "directional_auc", "qvalue", "tf"],
        ascending=[False, False, False, True, True],
    )
    top = evaluable.drop_duplicates("tf").reset_index(drop=True)
    top = top.drop(columns="direction_priority")
    top.insert(0, "discovery_rank", np.arange(1, len(top) + 1))
    return top.head(top_n).copy()


def summarize_top_markers_by_state(
    marker_df: pd.DataFrame,
    per_state_top_n: int = 5,
) -> pd.DataFrame:
    evaluable = marker_df[
        marker_df["evaluable"] & (marker_df["direction"] == "high_in_state")
    ].copy()
    evaluable = evaluable.sort_values(
        ["target_state", "separation_score", "directional_auc", "qvalue", "tf"],
        ascending=[True, False, False, True, True],
    )
    evaluable["state_rank"] = evaluable.groupby("target_state").cumcount() + 1
    return evaluable[evaluable["state_rank"] <= per_state_top_n].copy()


def _prepare_state_ranks(marker_df: pd.DataFrame) -> pd.DataFrame:
    evaluable = marker_df[marker_df["evaluable"]].copy()
    evaluable = evaluable.sort_values(
        ["target_state", "separation_score", "directional_auc", "qvalue", "tf"],
        ascending=[True, False, False, True, True],
    )
    evaluable["state_rank"] = evaluable.groupby("target_state").cumcount() + 1
    return evaluable


def build_validation_state_support(
    discovery_top_df: pd.DataFrame,
    validation_marker_tables: dict[str, pd.DataFrame],
    qvalue_threshold: float = 0.05,
    directional_auc_threshold: float = 0.65,
    top_rank_threshold: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prepared_tables = {
        study: _prepare_state_ranks(marker_df)
        for study, marker_df in validation_marker_tables.items()
    }

    score_rows: list[dict[str, float | int | str | bool]] = []
    summary_rows: list[dict[str, float | int | str | bool]] = []

    for row in discovery_top_df.to_dict(orient="records"):
        tf = str(row["tf"])
        target_state = str(row["target_state"])
        discovery_direction = str(row["direction"])
        discovery_sign = 1.0 if discovery_direction == "high_in_state" else -1.0
        score_row: dict[str, float | int | str | bool] = {
            "discovery_rank": int(row["discovery_rank"]),
            "tf": tf,
            "target_state": target_state,
            "discovery_direction": discovery_direction,
            "discovery_separation_score": float(row["separation_score"]),
            "discovery_directional_auc": float(row["directional_auc"]),
            "discovery_qvalue": float(row["qvalue"]),
        }
        evaluable_count = 0
        direction_preserved = 0
        support_count = 0
        top_rank_count = 0
        directional_auc_values: list[float] = []
        non_evaluable_studies: list[str] = []

        for study, marker_df in prepared_tables.items():
            hit = marker_df[
                (marker_df["tf"] == tf) & (marker_df["target_state"] == target_state)
            ]
            if hit.empty:
                non_evaluable_studies.append(study)
                score_row[f"{study}_evaluable"] = False
                score_row[f"{study}_direction"] = None
                score_row[f"{study}_directional_auc"] = float("nan")
                score_row[f"{study}_qvalue"] = float("nan")
                score_row[f"{study}_state_rank"] = float("nan")
                continue
            cohort_row = hit.iloc[0]
            cohort_auroc = float(cohort_row["auroc"])
            directional_auc_vs_discovery = (
                cohort_auroc if discovery_direction == "high_in_state" else 1.0 - cohort_auroc
            )
            same_direction = (
                np.sign(float(cohort_row["delta_mean"])) == discovery_sign
                if np.isfinite(float(cohort_row["delta_mean"]))
                else False
            )
            supported = (
                same_direction
                and float(cohort_row["qvalue"]) <= qvalue_threshold
                and directional_auc_vs_discovery >= directional_auc_threshold
            )
            top_ranked = same_direction and int(cohort_row["state_rank"]) <= top_rank_threshold
            evaluable_count += 1
            direction_preserved += int(same_direction)
            support_count += int(supported)
            top_rank_count += int(top_ranked)
            directional_auc_values.append(directional_auc_vs_discovery)
            score_row[f"{study}_evaluable"] = True
            score_row[f"{study}_direction"] = str(cohort_row["direction"])
            score_row[f"{study}_directional_auc"] = directional_auc_vs_discovery
            score_row[f"{study}_qvalue"] = float(cohort_row["qvalue"])
            score_row[f"{study}_state_rank"] = int(cohort_row["state_rank"])

        mean_external_auc = (
            float(np.mean(directional_auc_values)) if directional_auc_values else float("nan")
        )
        support_fraction = (
            float(support_count / evaluable_count) if evaluable_count else float("nan")
        )
        replication_status = "limited"
        if evaluable_count and support_count == evaluable_count:
            replication_status = "strong"
        elif evaluable_count and support_count >= max(2, int(np.ceil(evaluable_count / 2))):
            replication_status = "partial"

        summary_rows.append(
            {
                "discovery_rank": int(row["discovery_rank"]),
                "tf": tf,
                "target_state": target_state,
                "discovery_direction": discovery_direction,
                "discovery_directional_auc": float(row["directional_auc"]),
                "discovery_qvalue": float(row["qvalue"]),
                "total_external_datasets": len(prepared_tables),
                "evaluable_external_datasets": evaluable_count,
                "non_evaluable_external_datasets": len(non_evaluable_studies),
                "non_evaluable_external_studies": ", ".join(non_evaluable_studies),
                "direction_preserved_datasets": direction_preserved,
                "supportive_external_datasets": support_count,
                "top_rank_external_datasets": top_rank_count,
                "mean_external_directional_auc": mean_external_auc,
                "support_fraction": support_fraction,
                "supported_in_all_evaluable_datasets": (
                    bool(evaluable_count and support_count == evaluable_count)
                ),
                "supported_in_majority_evaluable_datasets": (
                    bool(evaluable_count and support_count >= int(np.ceil(evaluable_count / 2)))
                ),
                "replication_status": replication_status,
            }
        )
        score_rows.append(score_row)

    return pd.DataFrame(score_rows), pd.DataFrame(summary_rows)


def _analysis_alignment_verdict(
    support_summary: pd.DataFrame,
    legacy_validation_summary: pd.DataFrame | None,
) -> tuple[str, str, dict[str, object]]:
    strong_support = support_summary[support_summary["replication_status"] == "strong"][
        "tf"
    ].tolist()
    partial_support = support_summary[support_summary["replication_status"] == "partial"][
        "tf"
    ].tolist()
    strict_validated_tfs: list[str] = []
    if legacy_validation_summary is not None and not legacy_validation_summary.empty:
        strict_validated_tfs = legacy_validation_summary[
            legacy_validation_summary["validated_in_all_datasets"]
        ]["tf"].astype(str).tolist()

    basis = {
        "legacy_strict_validated_tfs": strict_validated_tfs,
        "exploratory_strong_support_tfs": strong_support,
        "exploratory_partial_support_tfs": partial_support,
    }
    if not strict_validated_tfs and not strong_support:
        return (
            "aligned",
            "기존 strict cross-cohort TF rule에서 validated TF가 없고, 이번 Wilcoxon "
            "marker screen에서도 strong external support TF가 없어 현재 TF layer "
            "hypothesis-generating framing과 일치한다.",
            basis,
        )
    if strict_validated_tfs:
        return (
            "tension",
            "기존 strict TF validation에서 이미 validated TF가 존재하므로 현재 analysis "
            "framing을 다시 점검해야 한다.",
            basis,
        )
    return (
        "mixed",
        "기존 strict TF validation은 여전히 약하지만, exploratory marker screen에서 일부 "
        "TF가 강한 external support를 보여 표현을 더 정교하게 다듬는 편이 안전하다.",
        basis,
    )


def write_state_separation_outputs(
    discovery_modeled_h5ad: str | Path,
    validation_root: str | Path,
    output_dir: str | Path,
    studies: tuple[str, ...] | None = None,
    top_n: int = 12,
    min_group_size: int = 20,
    qvalue_threshold: float = 0.05,
    directional_auc_threshold: float = 0.65,
    top_rank_threshold: int = 10,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    discovery_adata = read_h5ad(discovery_modeled_h5ad)
    discovery_markers = compute_tf_state_markers(discovery_adata, min_group_size=min_group_size)
    discovery_top = summarize_top_state_tfs(discovery_markers, top_n=top_n)
    discovery_by_state = summarize_top_markers_by_state(discovery_markers, per_state_top_n=5)

    validation_root_path = Path(validation_root)
    study_names = list(studies) if studies else sorted(
        path.name for path in validation_root_path.iterdir() if (path / "modeled.h5ad").exists()
    )

    validation_tables: dict[str, pd.DataFrame] = {}
    validation_long_frames: list[pd.DataFrame] = []
    for study in study_names:
        study_path = validation_root_path / study / "modeled.h5ad"
        marker_df = compute_tf_state_markers(read_h5ad(study_path), min_group_size=min_group_size)
        validation_tables[study] = marker_df
        marker_with_study = marker_df.copy()
        marker_with_study.insert(0, "dataset", study)
        validation_long_frames.append(marker_with_study)

    support_scores, support_summary = build_validation_state_support(
        discovery_top_df=discovery_top,
        validation_marker_tables=validation_tables,
        qvalue_threshold=qvalue_threshold,
        directional_auc_threshold=directional_auc_threshold,
        top_rank_threshold=top_rank_threshold,
    )
    support_summary = support_summary.sort_values("discovery_rank").reset_index(drop=True)

    discovery_markers.to_csv(output_path / "discovery_state_markers.csv", index=False)
    discovery_top.to_csv(output_path / "discovery_top_state_tfs.csv", index=False)
    discovery_by_state.to_csv(output_path / "discovery_top_markers_by_state.csv", index=False)
    pd.concat(validation_long_frames, ignore_index=True).to_csv(
        output_path / "validation_state_markers.csv",
        index=False,
    )
    support_scores.to_csv(output_path / "validation_state_support_scores.csv", index=False)
    support_summary.to_csv(output_path / "validation_state_support_summary.csv", index=False)

    legacy_validation_summary = None
    legacy_validation_path = validation_root_path / "validation_consensus_summary.csv"
    if legacy_validation_path.exists():
        legacy_validation_summary = pd.read_csv(legacy_validation_path)

    alignment_status, alignment_text, alignment_basis = _analysis_alignment_verdict(
        support_summary=support_summary,
        legacy_validation_summary=legacy_validation_summary,
    )
    payload = {
        "method": "one-vs-rest Wilcoxon marker analysis on TF activity",
        "discovery_modeled_h5ad": str(discovery_modeled_h5ad),
        "validation_root": str(validation_root),
        "studies": study_names,
        "top_n": top_n,
        "min_group_size": min_group_size,
        "qvalue_threshold": qvalue_threshold,
        "directional_auc_threshold": directional_auc_threshold,
        "top_rank_threshold": top_rank_threshold,
        "analysis_alignment": {
            "status": alignment_status,
            "summary": alignment_text,
            "basis": alignment_basis,
        },
        "top_discovery_tfs": support_summary.head(top_n).to_dict(orient="records"),
        "top_markers_by_state": {
            state: rows.drop(columns="target_state").to_dict(orient="records")
            for state, rows in discovery_by_state.groupby("target_state")
        },
    }
    (output_path / "state_separation_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# TF State Marker Summary",
        "",
        "- Method: one-vs-rest Wilcoxon marker analysis on TF activity",
        f"- Discovery file: {Path(discovery_modeled_h5ad)}",
        f"- Validation studies: {', '.join(study_names)}",
        f"- Minimum cells per one-vs-rest group: {min_group_size}",
        (
            "- Support rule: same state/direction, "
            f"q<={qvalue_threshold}, directional AUC>={directional_auc_threshold}, "
            f"top-{top_rank_threshold} tracked separately"
        ),
        f"- Paper alignment: {alignment_status}",
        f"- Alignment summary: {alignment_text}",
        (
            "- Legacy strict TF validation count: "
            f"{len(alignment_basis['legacy_strict_validated_tfs'])}"
        ),
        "",
        "## Discovery Top Markers By State",
        "",
    ]
    for state, rows in discovery_by_state.groupby("target_state"):
        top_terms = ", ".join(
            f"{row.tf} (AUC {row.directional_auc:.3f})" for row in rows.itertuples(index=False)
        )
        lines.append(f"- {state}: {top_terms}")

    lines.extend(
        [
            "",
            "## Global Discovery Leaderboard",
            "",
            (
                "- This global leaderboard is discovery-wide and is dominated by "
                "exhausted-anchor markers because those TF effects are much "
                "stronger than activated/boundary markers."
            ),
            "",
        ]
    )
    for row in support_summary.itertuples(index=False):
        skipped = (
            f"; skipped={row.non_evaluable_external_studies}"
            if row.non_evaluable_external_studies
            else ""
        )
        lines.append(
            (
                f"- {row.tf}: state={row.target_state}, discovery directional AUC="
                f"{row.discovery_directional_auc:.3f}, external support="
                f"{row.supportive_external_datasets}/{row.evaluable_external_datasets} "
                f"evaluable of {row.total_external_datasets} total, "
                f"mean external directional AUC={row.mean_external_directional_auc:.3f}, "
                f"status={row.replication_status}{skipped}"
            )
        )
    (output_path / "state_separation_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    return output_path
