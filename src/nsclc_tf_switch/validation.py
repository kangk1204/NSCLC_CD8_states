from __future__ import annotations

import json
import re
import subprocess
import tarfile
from gzip import decompress
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import io, sparse

from nsclc_tf_switch.config import DEFAULT_VALIDATION_STUDIES, PAPER_TF_PANEL
from nsclc_tf_switch.matrix_io import load_matrix_market_as_anndata
from nsclc_tf_switch.pipeline import prepare_anndata, run_graph_tf_analysis
from nsclc_tf_switch.reporting import save_validation_heatmap


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_gse171145_patient(sample_id: str) -> str:
    label = re.sub(r"^GSM\d+_", "", sample_id)
    label = re.sub(r"-T\d+$", "", label)
    label = re.sub(r"-T$", "", label)
    return label


def materialize_gse171145_tcells(raw_tar: str | Path, output_dir: str | Path) -> Path:
    output_path = Path(output_dir) / "GSE171145"
    output_path.mkdir(parents=True, exist_ok=True)

    sample_matrices: list[tuple[list[str], sparse.csc_matrix]] = []
    obs_frames: list[pd.DataFrame] = []
    feature_names: list[str] = []
    seen_genes: set[str] = set()

    with tarfile.open(raw_tar, "r") as tar:
        count_members = sorted(
            member.name for member in tar.getmembers() if member.name.endswith(".counts.tsv.gz")
        )
        for count_name in count_members:
            sample_id = count_name.replace(".counts.tsv.gz", "")
            cellname_name = count_name.replace(".counts.tsv.gz", ".cellname.list.txt.gz")

            count_bytes = tar.extractfile(count_name).read()
            cellname_bytes = tar.extractfile(cellname_name).read()

            count_df = pd.read_csv(BytesIO(decompress(count_bytes)), sep="\t")
            cellname_df = pd.read_csv(BytesIO(decompress(cellname_bytes)), sep="\t")
            cell_map = dict(
                zip(
                    cellname_df["CellIndex"].astype(str),
                    cellname_df["CellName"].astype(str),
                    strict=True,
                )
            )

            genes = count_df.iloc[:, 0].astype(str).tolist()
            for gene in genes:
                if gene not in seen_genes:
                    seen_genes.add(gene)
                    feature_names.append(gene)

            value_df = count_df.iloc[:, 1:].copy()
            value_df.columns = [cell_map.get(column, column) for column in value_df.columns]
            value_df.columns = [f"{sample_id}_{column}" for column in value_df.columns]
            sample_matrices.append((genes, sparse.csc_matrix(value_df.to_numpy(dtype=float))))

            obs_frames.append(
                pd.DataFrame(
                    {
                        "CellID": value_df.columns,
                        "sample_id": sample_id,
                        "Patient": _normalize_gse171145_patient(sample_id),
                        "Study": "GSE171145",
                        "Cell_Cluster_level1": "T",
                        "Cell_Cluster_level2": "T_unspecified",
                    }
                )
            )

    if not feature_names or not sample_matrices:
        raise ValueError("No count matrices found in GSE171145 tarball")

    global_index = {gene: idx for idx, gene in enumerate(feature_names)}
    aligned_matrices: list[sparse.csc_matrix] = []
    for genes, matrix in sample_matrices:
        local_index = {idx: global_index[gene] for idx, gene in enumerate(genes)}
        coo = matrix.tocoo()
        remapped = sparse.coo_matrix(
            (coo.data, ([local_index[row] for row in coo.row], coo.col)),
            shape=(len(feature_names), matrix.shape[1]),
        ).tocsc()
        aligned_matrices.append(remapped)

    merged = sparse.hstack(aligned_matrices, format="csc")
    io.mmwrite(output_path / "matrix.mtx", merged)
    pd.Series(feature_names).to_csv(
        output_path / "features.tsv",
        sep="\t",
        index=False,
        header=False,
    )
    pd.concat(obs_frames, ignore_index=True).to_csv(output_path / "obs.csv", index=False)
    return output_path


def materialize_validation_studies(
    raw_matrix_rds: str | Path,
    metadata_csv: str | Path,
    output_dir: str | Path,
    studies: tuple[str, ...] = DEFAULT_VALIDATION_STUDIES,
    cell_level1: str = "T",
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    script_path = _repo_root() / "scripts" / "extract_figshare_studies.R"
    command = [
        "Rscript",
        str(script_path),
        str(raw_matrix_rds),
        str(metadata_csv),
        str(output_path),
        cell_level1,
        *studies,
    ]
    subprocess.run(command, check=True)
    return output_path


def analyze_validation_study(
    study: str,
    materialized_dir: str | Path,
    output_dir: str | Path,
    epochs: int = 30,
) -> Path:
    study_dir = Path(materialized_dir) / study
    adata = load_matrix_market_as_anndata(
        matrix_path=study_dir / "matrix.mtx",
        features_path=study_dir / "features.tsv",
        obs_path=study_dir / "obs.csv",
        source_dataset=study,
    )
    adata.obs["Study"] = study
    adata = prepare_anndata(adata)
    return run_graph_tf_analysis(adata, output_dir=Path(output_dir) / study, epochs=epochs)


def _bootstrap_rho_ci(
    rhos: list[float], n_boot: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Return (mean_abs_rho, rho_ci_low, rho_ci_high) via percentile bootstrap.

    Returns NaNs when fewer than two finite rhos are supplied. Operates on |rho|
    because the caller wants the magnitude CI for the validation summary.
    """
    finite = np.asarray([r for r in rhos if r is not None and np.isfinite(r)], dtype=float)
    if finite.size < 2:
        mean_abs = float(np.mean(np.abs(finite))) if finite.size == 1 else float("nan")
        return mean_abs, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    abs_rhos = np.abs(finite)
    boot_means = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample = rng.choice(abs_rhos, size=abs_rhos.size, replace=True)
        boot_means[i] = float(sample.mean())
    lo = float(np.quantile(boot_means, alpha / 2))
    hi = float(np.quantile(boot_means, 1 - alpha / 2))
    return float(abs_rhos.mean()), lo, hi


def build_validation_consensus(
    discovery_ranking_path: str | Path,
    validation_root: str | Path,
    studies: tuple[str, ...] = DEFAULT_VALIDATION_STUDIES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    discovery = pd.read_csv(discovery_ranking_path)
    panel = list(dict.fromkeys([*PAPER_TF_PANEL, *discovery["tf"].head(10).tolist()]))

    score_rows: list[dict[str, float | str | None]] = []
    summary_rows: list[dict[str, float | str | int | bool | None]] = []

    rankings = {"discovery": discovery}
    for study in studies:
        rankings[study] = pd.read_csv(Path(validation_root) / study / "tf_switch_ranking.csv")

    for tf in panel:
        score_row: dict[str, float | str | None] = {"tf": tf}
        rho_signs: list[int] = []
        rho_values: list[float] = []
        external_rho_values: list[float] = []
        present_count = 0
        finite_count = 0
        discovery_sign: int | None = None
        for dataset_name, ranking in rankings.items():
            hit = ranking[ranking["tf"] == tf]
            rho = float(hit.iloc[0]["spearman_rho"]) if not hit.empty else None
            score = float(hit.iloc[0]["switch_score"]) if not hit.empty else None
            delta = float(hit.iloc[0]["delta_exhausted_vs_activated"]) if not hit.empty else None
            score_row[f"{dataset_name}_rho"] = rho
            score_row[f"{dataset_name}_switch_score"] = score
            score_row[f"{dataset_name}_delta"] = delta
            if rho is not None and pd.notna(rho) and delta is not None and pd.notna(delta):
                present_count += 1
                sign = 1 if rho >= 0 else -1
                rho_signs.append(sign)
                rho_values.append(float(rho))
                if dataset_name != "discovery":
                    external_rho_values.append(float(rho))
                finite_count += 1
                if dataset_name == "discovery":
                    discovery_sign = sign

        score_rows.append(score_row)
        consistent_with_discovery = 0
        if discovery_sign is not None:
            consistent_with_discovery = int(sum(sign == discovery_sign for sign in rho_signs))

        mean_abs, ci_low, ci_high = _bootstrap_rho_ci(external_rho_values)
        validated_in_all = bool(
            finite_count == len(rankings)
            and discovery_sign is not None
            and consistent_with_discovery == finite_count
        )

        summary_rows.append(
            {
                "tf": tf,
                "datasets_present": present_count,
                "finite_datasets": finite_count,
                "positive_rho_datasets": int(sum(sign > 0 for sign in rho_signs)),
                "negative_rho_datasets": int(sum(sign < 0 for sign in rho_signs)),
                "consistent_with_discovery_datasets": consistent_with_discovery,
                "mean_abs_rho": mean_abs,
                "rho_ci_low": ci_low,
                "rho_ci_high": ci_high,
                "direction_consensus": (
                    "positive"
                    if rho_signs and sum(rho_signs) > 0
                    else "negative"
                    if rho_signs and sum(rho_signs) < 0
                    else "mixed"
                ),
                "validated_in_all_datasets": validated_in_all,
            }
        )

    return pd.DataFrame(score_rows), pd.DataFrame(summary_rows)


def write_validation_consensus_outputs(
    discovery_ranking_path: str | Path,
    validation_root: str | Path,
    studies: tuple[str, ...],
) -> Path:
    output_path = Path(validation_root)
    score_df, summary_df = build_validation_consensus(
        discovery_ranking_path=discovery_ranking_path,
        validation_root=output_path,
        studies=studies,
    )
    score_df.to_csv(output_path / "validation_consensus_scores.csv", index=False)
    summary_df.to_csv(output_path / "validation_consensus_summary.csv", index=False)
    (output_path / "validation_summary.json").write_text(
        json.dumps(
            {
                "studies": list(studies),
                "validated_in_all_datasets": summary_df[
                    summary_df["validated_in_all_datasets"]
                ]["tf"].tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_validation_heatmap(score_df, output_path)
    return output_path


def run_validation_panel(
    discovery_ranking_path: str | Path,
    raw_matrix_rds: str | Path,
    metadata_csv: str | Path,
    output_dir: str | Path,
    studies: tuple[str, ...] = DEFAULT_VALIDATION_STUDIES,
    epochs: int = 30,
) -> Path:
    output_path = Path(output_dir)
    materialized_dir = output_path / "materialized"
    materialize_validation_studies(
        raw_matrix_rds=raw_matrix_rds,
        metadata_csv=metadata_csv,
        output_dir=materialized_dir,
        studies=studies,
    )

    for study in studies:
        analyze_validation_study(
            study=study,
            materialized_dir=materialized_dir,
            output_dir=output_path,
            epochs=epochs,
        )

    return write_validation_consensus_outputs(
        discovery_ranking_path=discovery_ranking_path,
        validation_root=output_path,
        studies=studies,
    )
