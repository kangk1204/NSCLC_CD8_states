from __future__ import annotations

import numpy as np
import pandas as pd
from anndata import AnnData

from nsclc_tf_switch.state_separation import (
    _analysis_alignment_verdict,
    build_validation_state_support,
    compute_tf_state_markers,
    summarize_top_markers_by_state,
    summarize_top_state_tfs,
)


def _synthetic_adata(seed: int = 0, weaken_boundary: bool = False) -> AnnData:
    rng = np.random.default_rng(seed)
    n_per_state = 30
    states = np.array(
        ["activated_anchor"] * n_per_state
        + ["transition_boundary"] * n_per_state
        + ["exhausted_anchor"] * n_per_state
    )
    index = [f"cell_{i}" for i in range(states.size)]

    act = np.concatenate(
        [
            rng.normal(2.0, 0.2, n_per_state),
            rng.normal(0.0, 0.2, n_per_state),
            rng.normal(-1.0, 0.2, n_per_state),
        ]
    )
    boundary_center = 2.0 if not weaken_boundary else 0.0
    boundary_sd = 0.2 if not weaken_boundary else 0.6
    boundary = np.concatenate(
        [
            rng.normal(0.0, 0.2, n_per_state),
            rng.normal(boundary_center, boundary_sd, n_per_state),
            rng.normal(0.0, 0.2, n_per_state),
        ]
    )
    exh = np.concatenate(
        [
            rng.normal(-1.0, 0.2, n_per_state),
            rng.normal(0.0, 0.2, n_per_state),
            rng.normal(2.0, 0.2, n_per_state),
        ]
    )
    noise = rng.normal(0.0, 1.0, states.size)

    adata = AnnData(X=np.zeros((states.size, 1), dtype=float))
    adata.obs_names = index
    adata.obs["transition_state"] = states
    adata.obsm["tf_activity"] = pd.DataFrame(
        {
            "ACT_TF": act,
            "BOUND_TF": boundary,
            "EXH_TF": exh,
            "NOISE_TF": noise,
        },
        index=index,
    )
    return adata


def test_compute_tf_state_markers_recovers_expected_states() -> None:
    marker_df = compute_tf_state_markers(_synthetic_adata(), min_group_size=10)
    top = summarize_top_state_tfs(marker_df, top_n=4).set_index("tf")

    assert top.loc["ACT_TF", "target_state"] == "activated_anchor"
    assert top.loc["BOUND_TF", "target_state"] == "transition_boundary"
    assert top.loc["EXH_TF", "target_state"] == "exhausted_anchor"
    assert top.loc["EXH_TF", "direction"] == "high_in_state"
    assert top.loc["EXH_TF", "qvalue"] < 0.05
    assert top.loc["EXH_TF", "directional_auc"] > 0.95


def test_build_validation_state_support_counts_external_support() -> None:
    discovery_top = summarize_top_state_tfs(
        compute_tf_state_markers(_synthetic_adata(seed=1), min_group_size=10),
        top_n=3,
    )
    validation_tables = {
        "cohort_a": compute_tf_state_markers(_synthetic_adata(seed=2), min_group_size=10),
        "cohort_b": compute_tf_state_markers(
            _synthetic_adata(seed=3, weaken_boundary=True),
            min_group_size=10,
        ),
    }
    _, summary_df = build_validation_state_support(
        discovery_top_df=discovery_top,
        validation_marker_tables=validation_tables,
        qvalue_threshold=0.05,
        directional_auc_threshold=0.65,
        top_rank_threshold=3,
    )
    by_tf = summary_df.set_index("tf")

    assert by_tf.loc["EXH_TF", "supportive_external_datasets"] == 2
    assert by_tf.loc["EXH_TF", "replication_status"] == "strong"
    assert by_tf.loc["BOUND_TF", "supportive_external_datasets"] < 2
    assert by_tf.loc["BOUND_TF", "replication_status"] == "limited"


def test_top_markers_by_state_and_alignment_context() -> None:
    marker_df = compute_tf_state_markers(_synthetic_adata(seed=4), min_group_size=10)
    by_state = summarize_top_markers_by_state(marker_df, per_state_top_n=2)
    assert set(by_state["target_state"]) == {
        "activated_anchor",
        "transition_boundary",
        "exhausted_anchor",
    }
    legacy_summary = pd.DataFrame(
        {
            "tf": ["EXH_TF", "ACT_TF"],
            "validated_in_all_datasets": [False, False],
        }
    )
    support_summary = pd.DataFrame(
        {
            "tf": ["EXH_TF", "ACT_TF"],
            "replication_status": ["limited", "limited"],
        }
    )
    status, _, basis = _analysis_alignment_verdict(support_summary, legacy_summary)
    assert status == "aligned"
    assert basis["legacy_strict_validated_tfs"] == []
    assert basis["exploratory_strong_support_tfs"] == []
