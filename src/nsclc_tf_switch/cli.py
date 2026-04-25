from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from nsclc_tf_switch.config import (
    DEFAULT_INTERACTION_VALIDATION_STUDIES,
    DEFAULT_VALIDATION_STUDIES,
)
from nsclc_tf_switch.data_access import download_named_dataset
from nsclc_tf_switch.interaction import analyze_allcell_interactions
from nsclc_tf_switch.interaction_validation import run_interaction_validation
from nsclc_tf_switch.loom_io import summarize_loom
from nsclc_tf_switch.metadata_tools import summarize_integrated_metadata
from nsclc_tf_switch.pipeline import analyze_loom
from nsclc_tf_switch.state_separation import write_state_separation_outputs
from nsclc_tf_switch.validation import (
    analyze_validation_study,
    materialize_gse171145_tcells,
    run_validation_panel,
    write_validation_consensus_outputs,
)

app = typer.Typer(add_completion=False, no_args_is_help=True)
DATA_RAW_DIR = Path("data/raw")
RESULTS_DIR = Path("results/run")
DISCOVERY_RESULTS_DIR = Path("results/full_ku_tcells")
VALIDATION_RESULTS_DIR = Path("results/validation_panel_6study")
INTERACTION_RESULTS_DIR = Path("results/interactions")
INTERACTION_VALIDATION_RESULTS_DIR = Path("results/interaction_validation_5study")
STATE_SEPARATION_RESULTS_DIR = Path("results/state_separation_6study")


@app.command("download")
def download_command(
    dataset: Annotated[str, typer.Option("--dataset", "-d")],
    output_dir: Annotated[Path, typer.Option("--output-dir")] = DATA_RAW_DIR,
) -> None:
    path = download_named_dataset(dataset, output_dir)
    typer.echo(str(path))


@app.command("inspect-loom")
def inspect_loom_command(path: Path) -> None:
    summary = summarize_loom(path)
    typer.echo(json.dumps(summary, indent=2))


@app.command("analyze-loom")
def analyze_loom_command(
    path: Path,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = RESULTS_DIR,
    max_cells: Annotated[int | None, typer.Option("--max-cells")] = None,
    epochs: Annotated[int, typer.Option("--epochs")] = 40,
) -> None:
    destination = analyze_loom(path=path, output_dir=output_dir, max_cells=max_cells, epochs=epochs)
    typer.echo(str(destination))


@app.command("summarize-metadata")
def summarize_metadata_command(path: Path) -> None:
    summary = summarize_integrated_metadata(path)
    typer.echo(summary.to_json(orient="records", indent=2))


@app.command("run-validation-panel")
def run_validation_panel_command(
    discovery_ranking: Annotated[Path, typer.Option("--discovery-ranking")],
    raw_matrix_rds: Annotated[Path, typer.Option("--raw-matrix-rds")] = (
        DATA_RAW_DIR / "figshare_rawcounts" / "RNA_rawcounts_matrix.rds"
    ),
    metadata_csv: Annotated[Path, typer.Option("--metadata-csv")] = (
        DATA_RAW_DIR / "figshare_metadata" / "metadata.csv"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = RESULTS_DIR / "validation_panel",
    studies: Annotated[list[str] | None, typer.Option("--study")] = None,
    epochs: Annotated[int, typer.Option("--epochs")] = 30,
) -> None:
    destination = run_validation_panel(
        discovery_ranking_path=discovery_ranking,
        raw_matrix_rds=raw_matrix_rds,
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        studies=tuple(studies) if studies else DEFAULT_VALIDATION_STUDIES,
        epochs=epochs,
    )
    typer.echo(str(destination))


@app.command("analyze-interactions")
def analyze_interactions_command(
    loom_path: Path,
    output_dir: Annotated[Path, typer.Option("--output-dir")] = RESULTS_DIR / "interactions",
    epochs: Annotated[int, typer.Option("--epochs")] = 25,
) -> None:
    destination = analyze_allcell_interactions(
        loom_path=loom_path,
        output_dir=output_dir,
        epochs=epochs,
    )
    typer.echo(str(destination))


@app.command("run-interaction-validation")
def run_interaction_validation_command(
    discovery_interaction: Annotated[Path, typer.Option("--discovery-interaction")] = (
        INTERACTION_RESULTS_DIR / "interaction_network.csv"
    ),
    raw_matrix_rds: Annotated[Path, typer.Option("--raw-matrix-rds")] = (
        DATA_RAW_DIR / "figshare_rawcounts" / "RNA_rawcounts_matrix.rds"
    ),
    metadata_csv: Annotated[Path, typer.Option("--metadata-csv")] = (
        DATA_RAW_DIR / "figshare_metadata" / "metadata.csv"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = (
        INTERACTION_VALIDATION_RESULTS_DIR
    ),
    studies: Annotated[list[str] | None, typer.Option("--study")] = None,
    epochs: Annotated[int, typer.Option("--epochs")] = 25,
) -> None:
    destination = run_interaction_validation(
        discovery_interaction_path=discovery_interaction,
        raw_matrix_rds=raw_matrix_rds,
        metadata_csv=metadata_csv,
        output_dir=output_dir,
        studies=tuple(studies) if studies else DEFAULT_INTERACTION_VALIDATION_STUDIES,
        epochs=epochs,
    )
    typer.echo(str(destination))


@app.command("materialize-gse171145")
def materialize_gse171145_command(
    raw_tar: Annotated[Path, typer.Option("--raw-tar")] = (
        DATA_RAW_DIR / "gse171145" / "GSE171145_RAW.tar"
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = (
        RESULTS_DIR / "gse171145_materialized"
    ),
) -> None:
    destination = materialize_gse171145_tcells(raw_tar=raw_tar, output_dir=output_dir)
    typer.echo(str(destination))


@app.command("analyze-validation-study")
def analyze_validation_study_command(
    study: Annotated[str, typer.Option("--study")],
    materialized_dir: Annotated[Path, typer.Option("--materialized-dir")],
    output_dir: Annotated[Path, typer.Option("--output-dir")],
    epochs: Annotated[int, typer.Option("--epochs")] = 30,
) -> None:
    destination = analyze_validation_study(
        study=study,
        materialized_dir=materialized_dir,
        output_dir=output_dir,
        epochs=epochs,
    )
    typer.echo(str(destination))


@app.command("build-validation-consensus")
def build_validation_consensus_command(
    discovery_ranking: Annotated[Path, typer.Option("--discovery-ranking")],
    validation_root: Annotated[Path, typer.Option("--validation-root")],
    studies: Annotated[list[str], typer.Option("--study")],
) -> None:
    destination = write_validation_consensus_outputs(
        discovery_ranking_path=discovery_ranking,
        validation_root=validation_root,
        studies=tuple(studies),
    )
    typer.echo(str(destination))


@app.command("run-state-separation-analysis")
def run_state_separation_analysis_command(
    discovery_modeled: Annotated[Path, typer.Option("--discovery-modeled")] = (
        DISCOVERY_RESULTS_DIR / "modeled.h5ad"
    ),
    validation_root: Annotated[Path, typer.Option("--validation-root")] = (
        VALIDATION_RESULTS_DIR
    ),
    output_dir: Annotated[Path, typer.Option("--output-dir")] = STATE_SEPARATION_RESULTS_DIR,
    studies: Annotated[list[str] | None, typer.Option("--study")] = None,
    top_n: Annotated[int, typer.Option("--top-n")] = 12,
    min_group_size: Annotated[int, typer.Option("--min-group-size")] = 20,
    qvalue_threshold: Annotated[float, typer.Option("--qvalue-threshold")] = 0.05,
    directional_auc_threshold: Annotated[
        float, typer.Option("--directional-auc-threshold")
    ] = 0.65,
    top_rank_threshold: Annotated[int, typer.Option("--top-rank-threshold")] = 10,
) -> None:
    destination = write_state_separation_outputs(
        discovery_modeled_h5ad=discovery_modeled,
        validation_root=validation_root,
        output_dir=output_dir,
        studies=tuple(studies) if studies else None,
        top_n=top_n,
        min_group_size=min_group_size,
        qvalue_threshold=qvalue_threshold,
        directional_auc_threshold=directional_auc_threshold,
        top_rank_threshold=top_rank_threshold,
    )
    typer.echo(str(destination))


if __name__ == "__main__":
    app()
