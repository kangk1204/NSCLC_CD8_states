from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

from nsclc_tf_switch.tf_activity import benjamini_hochberg, rank_tf_switches


@pytest.fixture()
def synthetic_adata() -> AnnData:
    rng = np.random.default_rng(0)
    n_cells = 180
    probabilities = np.concatenate([
        rng.uniform(0.0, 0.2, 60),
        rng.uniform(0.4, 0.6, 60),
        rng.uniform(0.8, 1.0, 60),
    ])
    states = np.array(
        ["activated_anchor"] * 60 + ["transition_boundary"] * 60 + ["exhausted_anchor"] * 60
    )

    # TF "UP" increases monotonically with exhaustion probability.
    tf_up = probabilities + rng.normal(0, 0.05, n_cells)
    # TF "DOWN" decreases with probability.
    tf_down = (1.0 - probabilities) + rng.normal(0, 0.05, n_cells)
    # TF "NULL" is uninformative noise.
    tf_null = rng.normal(0, 1.0, n_cells)

    tf_scores = pd.DataFrame(
        {"UP": tf_up, "DOWN": tf_down, "NULL": tf_null},
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    X = rng.normal(0, 1, size=(n_cells, 2))
    adata = AnnData(X=X)
    adata.obs_names = tf_scores.index
    adata.obs["transition_probability"] = probabilities
    adata.obs["transition_state"] = states
    adata.obsm["tf_activity"] = tf_scores
    return adata


def test_rank_tf_switches_preserves_direction(synthetic_adata: AnnData) -> None:
    ranking = rank_tf_switches(synthetic_adata, top_n=10)

    expected_columns = {
        "tf",
        "spearman_rho",
        "spearman_pvalue",
        "spearman_qvalue",
        "delta_exhausted_vs_activated",
        "boundary_shift",
        "switch_score",
        "switch_score_signed",
    }
    assert expected_columns.issubset(ranking.columns)

    by_tf = ranking.set_index("tf")
    assert by_tf.loc["UP", "switch_score_signed"] > 0
    assert by_tf.loc["DOWN", "switch_score_signed"] < 0
    # Signed magnitude never exceeds unsigned magnitude.
    assert np.allclose(
        ranking["switch_score"], np.abs(ranking["switch_score_signed"]), atol=1e-9
    )


def test_rank_tf_switches_qvalues_ordered(synthetic_adata: AnnData) -> None:
    ranking = rank_tf_switches(synthetic_adata, top_n=3)
    finite = ranking.dropna(subset=["spearman_qvalue", "spearman_pvalue"])
    # BH q-values are monotonically non-decreasing when sorted by raw p-value.
    ordered = finite.sort_values("spearman_pvalue")
    q = ordered["spearman_qvalue"].to_numpy()
    assert np.all(q[:-1] <= q[1:] + 1e-12)
    # q-values sit within [0, 1].
    assert (q >= 0).all() and (q <= 1).all()
    # UP and DOWN should be significant on the synthetic toy.
    by_tf = ranking.set_index("tf")
    assert by_tf.loc["UP", "spearman_qvalue"] < 0.1
    assert by_tf.loc["DOWN", "spearman_qvalue"] < 0.1


def test_benjamini_hochberg_matches_manual() -> None:
    pvalues = np.array([0.01, 0.04, 0.03, 0.005, np.nan])
    q = benjamini_hochberg(pvalues)
    assert np.isnan(q[-1])
    finite_q = q[:-1]
    # All q-values <= 1 and >= the raw p-value when only 4 tests applied.
    assert (finite_q <= 1.0).all()
    assert (finite_q >= pvalues[:-1]).all()
    # Smallest p-value (0.005) should achieve the minimum q (ties are allowed).
    assert finite_q[3] == finite_q.min()
