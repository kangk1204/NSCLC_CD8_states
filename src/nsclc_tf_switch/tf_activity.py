from __future__ import annotations

import decoupler as dc
import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import spearmanr


def score_tf_activity(adata: AnnData, levels: tuple[str, ...] = ("A", "B", "C")) -> pd.DataFrame:
    network = dc.get_dorothea(organism="human", levels=list(levels))
    working = adata.copy()
    working.var_names = pd.Index(
        [str(name).upper() for name in working.var_names],
        name="gene_symbol",
    )
    dc.run_ulm(working, network, source="source", target="target", weight="weight", use_raw=False)
    estimates = working.obsm["ulm_estimate"]
    if isinstance(estimates, pd.DataFrame):
        tf_scores = estimates.copy()
    else:
        tf_scores = pd.DataFrame(estimates, index=working.obs_names)
        tf_scores.columns = working.uns.get("ulm_sources", tf_scores.columns)
    adata.obsm["tf_activity"] = tf_scores
    return tf_scores


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Return Benjamini-Hochberg adjusted q-values. NaNs propagate through unchanged."""
    p = np.asarray(pvalues, dtype=float)
    out = np.full_like(p, np.nan, dtype=float)
    finite_mask = np.isfinite(p)
    if not finite_mask.any():
        return out
    finite_p = p[finite_mask]
    n = finite_p.size
    order = np.argsort(finite_p)
    ranked = finite_p[order]
    adjusted = ranked * n / (np.arange(n) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q = np.empty(n, dtype=float)
    q[order] = adjusted
    out[finite_mask] = q
    return out


def rank_tf_switches(adata: AnnData, top_n: int = 25) -> pd.DataFrame:
    tf_scores = adata.obsm["tf_activity"]
    probabilities = adata.obs["transition_probability"].to_numpy()
    states = adata.obs["transition_state"].astype(str)

    activated = states == "activated_anchor"
    boundary = states == "transition_boundary"
    exhausted = states == "exhausted_anchor"

    rows: list[dict[str, float | str]] = []
    for tf in tf_scores.columns:
        values = tf_scores[tf].to_numpy()
        rho, pvalue = spearmanr(values, probabilities)
        rho = float(rho) if rho is not None and np.isfinite(rho) else float("nan")
        pvalue = float(pvalue) if pvalue is not None and np.isfinite(pvalue) else float("nan")
        delta = float(values[exhausted].mean() - values[activated].mean())
        activated_mean = values[activated].mean()
        exhausted_mean = values[exhausted].mean()
        boundary_shift = float(values[boundary].mean() - 0.5 * (activated_mean + exhausted_mean))
        magnitude = abs(rho) * abs(delta) * (1.0 + abs(boundary_shift))
        direction = np.sign(rho) if np.isfinite(rho) and rho != 0 else np.sign(delta)
        switch_score_signed = float(direction * magnitude)
        rows.append(
            {
                "tf": str(tf),
                "spearman_rho": rho,
                "spearman_pvalue": pvalue,
                "delta_exhausted_vs_activated": delta,
                "boundary_shift": boundary_shift,
                "switch_score": float(magnitude),
                "switch_score_signed": switch_score_signed,
            }
        )

    ranking = pd.DataFrame(rows)
    ranking["spearman_qvalue"] = benjamini_hochberg(ranking["spearman_pvalue"].to_numpy())
    ranking = ranking.sort_values("switch_score", ascending=False).reset_index(drop=True)
    return ranking.head(top_n)
