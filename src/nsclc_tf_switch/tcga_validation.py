"""Independent TCGA NSCLC bulk RNA-seq validation for the recurrent anchor edges.

Provides:
- download_tcga_expression(): cache TCGA-LUAD + TCGA-LUSC gene-expression matrices from
  the classic UCSC Xena TCGA hub (``tcga.xenahubs.net``) as HGNC-symbol-indexed parquet
  files. Illumina HiSeqV2 RNA-seq, normalised to log2(RSEM+1).
- compute_anchor_edge_tcga_validation(): compute per-cohort Spearman ρ between each
  anchor edge ligand×receptor co-expression score and a CD8 exhaustion signature.

The exhaustion signature is the mean per-sample z-score of PDCD1, CTLA4, LAG3, TIGIT,
HAVCR2, TOX (canonical CD8 exhaustion markers). The ligand×receptor co-expression score
is the arithmetic mean of the two log2(RSEM+1) expression values (equivalent to the
geometric mean in linear space). These are bulk proxies, not sender-resolved
sc interaction scores, and are reported as such.
"""

from __future__ import annotations

import gzip
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Xena TCGA hub URLs (log2(RSEM+1), HGNC-symbol-indexed).
XENA_LUAD_URL = "https://tcga.xenahubs.net/download/TCGA.LUAD.sampleMap/HiSeqV2.gz"
XENA_LUSC_URL = "https://tcga.xenahubs.net/download/TCGA.LUSC.sampleMap/HiSeqV2.gz"

EXHAUSTION_SIGNATURE_GENES: tuple[str, ...] = (
    "PDCD1", "CTLA4", "LAG3", "TIGIT", "HAVCR2", "TOX",
)

ANCHOR_EDGES: tuple[tuple[str, str, str, str], ...] = (
    # (edge_name, ligand, receptor, sender_group_hint)
    ("b_plasma_CD274_PDCD1", "CD274", "PDCD1", "b_plasma"),
    ("cancer_CD274_PDCD1", "CD274", "PDCD1", "cancer"),
    ("treg_LGALS9_HAVCR2", "LGALS9", "HAVCR2", "treg"),
    ("b_plasma_CD80_CTLA4", "CD80", "CTLA4", "b_plasma"),
)
# A non-anchor control edge included in the BH FDR panel so the correction is not trivial.
CONTROL_EDGE: tuple[str, str, str, str] = ("cancer_CD86_CTLA4", "CD86", "CTLA4", "cancer")


def _http_get(url: str, timeout_s: int = 600) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "nsclc-tf-switch/0.1"})
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        return response.read()


def _expression_to_symbol_matrix(expression_path: Path) -> pd.DataFrame:
    """Return a (gene_symbol × sample) float32 DataFrame. Xena HiSeqV2 files are already
    HGNC-symbol indexed, so the step is just a gzipped TSV read."""
    with gzip.open(expression_path, "rt") as handle:
        matrix = pd.read_csv(handle, sep="\t", index_col=0)
    matrix.index = matrix.index.astype(str)
    matrix.index.name = "gene_symbol"
    return matrix.astype(np.float32)


def download_tcga_expression(
    output_dir: str | Path = "data/raw/tcga",
    cohorts: tuple[str, ...] = ("LUAD", "LUSC"),
    force: bool = False,
) -> dict[str, Path]:
    """Download and cache LUAD + LUSC expression matrices as parquet.

    Writes ``TCGA-{cohort}.symbol_fpkm.parquet`` under ``output_dir`` and returns a
    mapping from cohort name to parquet Path. The filename keeps the ``fpkm`` suffix for
    backward compatibility with the rest of the pipeline, but the underlying units are
    log2(RSEM+1) from Xena's Illumina HiSeqV2 resource.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    urls = {"LUAD": XENA_LUAD_URL, "LUSC": XENA_LUSC_URL}
    results: dict[str, Path] = {}
    for cohort in cohorts:
        if cohort not in urls:
            raise ValueError(f"unknown TCGA cohort: {cohort}")
        expression_path = output_path / f"TCGA-{cohort}.HiSeqV2.gz"
        parquet_path = output_path / f"TCGA-{cohort}.symbol_fpkm.parquet"
        if force or not parquet_path.exists():
            if force or not expression_path.exists() or expression_path.stat().st_size < 1_000:
                expression_path.write_bytes(_http_get(urls[cohort]))
            matrix = _expression_to_symbol_matrix(expression_path)
            matrix.to_parquet(parquet_path)
        results[cohort] = parquet_path
    return results


def _compute_exhaustion_signature(
    expression: pd.DataFrame, genes: tuple[str, ...] = EXHAUSTION_SIGNATURE_GENES
) -> pd.Series:
    missing = [g for g in genes if g not in expression.index]
    if missing:
        raise KeyError(f"Missing exhaustion signature genes: {missing}")
    block = expression.loc[list(genes)].astype(float)
    zscored = block.sub(block.mean(axis=1), axis=0).div(
        block.std(axis=1, ddof=0).replace(0, np.nan), axis=0
    )
    return zscored.mean(axis=0).rename("exhaustion_signature")


def _coexpression_score(
    expression: pd.DataFrame, ligand: str, receptor: str
) -> pd.Series:
    if ligand not in expression.index or receptor not in expression.index:
        raise KeyError(f"Missing edge genes: ligand={ligand} receptor={receptor}")
    # Xena matrices are log2(RSEM+1); arithmetic mean of logs = geometric mean in linear.
    return expression.loc[[ligand, receptor]].mean(axis=0).rename(
        f"{ligand}_x_{receptor}"
    )


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    mask = np.isfinite(p)
    if not mask.any():
        return out
    finite = p[mask]
    n = finite.size
    order = np.argsort(finite)
    ranked = finite[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q = np.empty(n, dtype=float)
    q[order] = adjusted
    out[mask] = q
    return out


def _bootstrap_rho_ci(
    x: np.ndarray, y: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = x.size
    if n < 10:
        return float("nan"), float("nan")
    rhos = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        r, _ = spearmanr(x[idx], y[idx])
        rhos[i] = r if r is not None and np.isfinite(r) else 0.0
    return float(np.quantile(rhos, 0.025)), float(np.quantile(rhos, 0.975))


def compute_anchor_edge_tcga_validation(
    parquet_paths: dict[str, str | Path],
    output_csv: str | Path = "results/tcga_validation/tcga_anchor_edge_correlations.csv",
    scatter_output_csv: str | Path | None = (
        "results/tcga_validation/tcga_anchor_edge_scatter.csv"
    ),
    edges: tuple[tuple[str, str, str, str], ...] = ANCHOR_EDGES + (CONTROL_EDGE,),
) -> Path:
    """Compute per-(cohort, edge) Spearman ρ between ligand×receptor co-expression and
    the CD8 exhaustion signature. Writes the CSV and returns the path."""
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scatter_rows: list[dict[str, float | str]] = []
    all_rows: list[dict[str, float | str]] = []

    for cohort, path in parquet_paths.items():
        expression = pd.read_parquet(path)
        exhaustion = _compute_exhaustion_signature(expression)
        cohort_rows: list[dict[str, float | str]] = []
        for edge_name, ligand, receptor, sender in edges:
            try:
                coexpression = _coexpression_score(expression, ligand, receptor)
            except KeyError:
                cohort_rows.append(
                    {
                        "cohort": cohort,
                        "edge": edge_name,
                        "sender_group_hint": sender,
                        "ligand": ligand,
                        "receptor": receptor,
                        "n_samples": 0,
                        "spearman_rho": float("nan"),
                        "spearman_pvalue": float("nan"),
                        "rho_ci_low": float("nan"),
                        "rho_ci_high": float("nan"),
                    }
                )
                continue
            aligned = pd.concat([coexpression, exhaustion], axis=1).dropna()
            rho, pvalue = spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
            ci_low, ci_high = _bootstrap_rho_ci(
                aligned.iloc[:, 0].to_numpy(),
                aligned.iloc[:, 1].to_numpy(),
            )
            cohort_rows.append(
                {
                    "cohort": cohort,
                    "edge": edge_name,
                    "sender_group_hint": sender,
                    "ligand": ligand,
                    "receptor": receptor,
                    "n_samples": int(aligned.shape[0]),
                    "spearman_rho": float(rho) if rho is not None else float("nan"),
                    "spearman_pvalue": float(pvalue) if pvalue is not None else float("nan"),
                    "rho_ci_low": ci_low,
                    "rho_ci_high": ci_high,
                }
            )
            for sample_id, values in aligned.iterrows():
                scatter_rows.append(
                    {
                        "cohort": cohort,
                        "edge": edge_name,
                        "sample_id": sample_id,
                        "coexpression": float(values.iloc[0]),
                        "exhaustion_signature": float(values.iloc[1]),
                    }
                )
        p_values = np.array([r["spearman_pvalue"] for r in cohort_rows])
        q_values = _benjamini_hochberg(p_values)
        for row, q in zip(cohort_rows, q_values, strict=True):
            row["spearman_qvalue"] = float(q) if np.isfinite(q) else float("nan")
        all_rows.extend(cohort_rows)

    result = pd.DataFrame(all_rows)
    result.to_csv(output_path, index=False)
    if scatter_output_csv is not None and scatter_rows:
        scatter_path = Path(scatter_output_csv)
        scatter_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(scatter_rows).to_csv(scatter_path, index=False)
    return output_path
