from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from anndata import AnnData


def _prepare_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_transition_embedding(adata: AnnData, output_dir: Path) -> Path:
    output_dir = _prepare_output_dir(output_dir)
    latent = adata.obsm["X_latent"]
    plot_df = pd.DataFrame(
        {
            "latent_1": latent[:, 0],
            "latent_2": latent[:, 1] if latent.shape[1] > 1 else 0.0,
            "transition_probability": adata.obs["transition_probability"].to_numpy(),
            "transition_state": adata.obs["transition_state"].astype(str).to_numpy(),
        }
    )
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=plot_df,
        x="latent_1",
        y="latent_2",
        hue="transition_probability",
        palette="viridis",
        s=10,
        linewidth=0,
    )
    plt.title("Graph latent embedding colored by transition probability")
    plt.tight_layout()
    destination = output_dir / "transition_embedding.png"
    plt.savefig(destination, dpi=200)
    plt.close()
    return destination


def save_top_tf_boxplot(
    adata: AnnData,
    ranking: pd.DataFrame,
    output_dir: Path,
    n_top: int = 12,
) -> Path:
    output_dir = _prepare_output_dir(output_dir)
    tf_scores = adata.obsm["tf_activity"][ranking["tf"].head(n_top)]
    transition_state = adata.obs["transition_state"].astype(str).to_numpy()
    long_df = tf_scores.assign(transition_state=transition_state).melt(
        id_vars="transition_state", var_name="tf", value_name="activity"
    )
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=long_df, x="tf", y="activity", hue="transition_state", showfliers=False)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top TF activities across inferred transition states")
    plt.tight_layout()
    destination = output_dir / "top_tf_boxplot.png"
    plt.savefig(destination, dpi=200)
    plt.close()
    return destination


def save_run_summary(adata: AnnData, ranking: pd.DataFrame, output_dir: Path) -> Path:
    output_dir = _prepare_output_dir(output_dir)
    summary = {
        "n_cells": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "state_counts": adata.obs["transition_state"].astype(str).value_counts().to_dict(),
        "top_tf": ranking["tf"].head(10).tolist(),
    }
    destination = output_dir / "run_summary.json"
    destination.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return destination


def save_validation_heatmap(score_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir = _prepare_output_dir(output_dir)
    rho_columns = [column for column in score_df.columns if column.endswith("_rho")]
    heatmap_df = score_df.set_index("tf")[rho_columns].dropna(how="all")
    plt.figure(figsize=(8, max(4, len(heatmap_df) * 0.35)))
    sns.heatmap(heatmap_df, cmap="coolwarm", center=0.0, linewidths=0.5)
    plt.title("Validation-panel TF direction consistency")
    plt.tight_layout()
    destination = output_dir / "validation_heatmap.png"
    plt.savefig(destination, dpi=200)
    plt.close()
    return destination


def save_interaction_heatmap(
    interaction_df: pd.DataFrame,
    output_dir: Path,
    top_n: int = 20,
) -> Path:
    output_dir = _prepare_output_dir(output_dir)
    working = interaction_df.copy()
    working["interaction"] = working["ligand"] + "->" + working["receptor"]
    top_labels = (
        working.groupby("interaction")["delta_score"]
        .max()
        .sort_values(ascending=False)
        .head(top_n)
        .index
    )
    pivot = working[working["interaction"].isin(top_labels)].pivot_table(
        index="interaction",
        columns="sender_group",
        values="delta_score",
        fill_value=0.0,
    )
    plt.figure(figsize=(9, max(5, len(pivot) * 0.35)))
    sns.heatmap(pivot, cmap="mako", center=0.0, linewidths=0.5)
    plt.title("Exhausted-vs-activated interaction delta")
    plt.tight_layout()
    destination = output_dir / "interaction_heatmap.png"
    plt.savefig(destination, dpi=200)
    plt.close()
    return destination


def save_patient_association_plot(association_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir = _prepare_output_dir(output_dir)
    plt.figure(figsize=(7, 4))
    sns.barplot(data=association_df, x="sender_group", y="spearman_rho", color="#4c7b8b")
    plt.xticks(rotation=30, ha="right")
    plt.title("Sender-group association with exhausted CD8 burden")
    plt.tight_layout()
    destination = output_dir / "patient_association_barplot.png"
    plt.savefig(destination, dpi=200)
    plt.close()
    return destination


def save_interaction_validation_heatmap(score_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir = _prepare_output_dir(output_dir)
    delta_columns = [column for column in score_df.columns if column.endswith("_delta")]
    heatmap_df = (
        score_df.assign(
            edge=lambda df: df["sender_group"] + ":" + df["ligand"] + "->" + df["receptor"]
        )
        .set_index("edge")[delta_columns]
        .dropna(how="all")
        .head(30)
    )
    plt.figure(figsize=(9, max(5, len(heatmap_df) * 0.35)))
    sns.heatmap(heatmap_df, cmap="coolwarm", center=0.0, linewidths=0.5)
    plt.title("Cross-cohort interaction delta consistency")
    plt.tight_layout()
    destination = output_dir / "interaction_validation_heatmap.png"
    plt.savefig(destination, dpi=200)
    plt.close()
    return destination
