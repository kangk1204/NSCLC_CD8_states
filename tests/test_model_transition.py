from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData

from nsclc_tf_switch.model import infer_transition_states


def _make_adata(delta: np.ndarray, latent: np.ndarray) -> AnnData:
    n = delta.size
    assert latent.shape[0] == n
    adata = AnnData(X=np.zeros((n, 2), dtype=np.float32))
    adata.obs_names = [f"c{i}" for i in range(n)]
    adata.obs["marker_delta"] = delta
    adata.obsm["X_latent"] = latent.astype(np.float32)
    return adata


def test_infer_transition_states_separable_classifier_path() -> None:
    """Classifier can separate two well-defined clusters and emit both anchor classes."""
    rng = np.random.default_rng(0)
    n = 200
    # Two halves: low delta → activated; high delta → exhausted.
    delta = np.concatenate([rng.uniform(-3.0, -1.0, n // 2), rng.uniform(1.0, 3.0, n // 2)])
    # Latent coordinates mirror the delta cluster so the classifier has real signal.
    latent_low = rng.normal(loc=(-4.0, 0.0), scale=0.3, size=(n // 2, 2))
    latent_high = rng.normal(loc=(4.0, 0.0), scale=0.3, size=(n // 2, 2))
    latent = np.vstack([latent_low, latent_high])

    adata = _make_adata(delta, latent)
    result = infer_transition_states(adata)

    probs = result["transition_probability"].to_numpy()
    states = result["transition_state"].astype(str)

    # All probabilities stay in [0, 1].
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # Both anchor classes are present.
    activated_fraction = float((states == "activated_anchor").mean())
    exhausted_fraction = float((states == "exhausted_anchor").mean())
    assert activated_fraction > 0.30
    assert exhausted_fraction > 0.30

    # Direction check: exhausted anchors should be concentrated where delta is high.
    exhausted_mask = states == "exhausted_anchor"
    activated_mask = states == "activated_anchor"
    assert delta[exhausted_mask].mean() > delta[activated_mask].mean()


def test_infer_transition_states_fallback_path() -> None:
    """When the classifier cannot distinguish the two anchor classes, the marker-delta
    quantile fallback should still produce both anchor categories and valid probs."""
    rng = np.random.default_rng(1)
    n = 200
    delta = rng.uniform(-2.0, 2.0, n)
    # Latent features are pure noise — no relationship to delta, so the classifier
    # returns probabilities near 0.5 and cannot exceed the 0.8 / 0.2 cutoffs.
    latent = rng.normal(0.0, 1e-4, size=(n, 2))

    adata = _make_adata(delta, latent)
    result = infer_transition_states(adata)

    probs = result["transition_probability"].to_numpy()
    states = result["transition_state"].astype(str)

    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    # Fallback assigns anchors via quantile cuts of delta, so both classes exist.
    assert (states == "activated_anchor").any()
    assert (states == "exhausted_anchor").any()
    assert (states == "transition_boundary").any()

    # Activated anchors sit at low delta; exhausted anchors at high delta.
    activated = states == "activated_anchor"
    exhausted = states == "exhausted_anchor"
    assert delta[activated].max() < delta[exhausted].min()


def test_infer_transition_states_probabilities_monotonic_in_delta() -> None:
    """On the separable case, mean transition probability should be higher for the
    high-delta half than for the low-delta half."""
    rng = np.random.default_rng(2)
    n = 120
    delta = np.concatenate([np.full(n // 2, -2.0), np.full(n // 2, 2.0)]) + rng.normal(0, 0.05, n)
    latent = np.column_stack([delta, rng.normal(0, 0.05, n)])
    adata = _make_adata(delta, latent)
    result = infer_transition_states(adata)

    probs = result["transition_probability"].to_numpy()
    assert probs[: n // 2].mean() < probs[n // 2 :].mean()


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_infer_transition_states_is_deterministic(seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = 150
    delta = np.concatenate([rng.uniform(-3, -1, n // 2), rng.uniform(1, 3, n // 2)])
    latent = np.column_stack([delta + rng.normal(0, 0.1, n), rng.normal(0, 0.1, n)])
    a = _make_adata(delta.copy(), latent.copy())
    b = _make_adata(delta.copy(), latent.copy())
    r1 = infer_transition_states(a)
    r2 = infer_transition_states(b)
    np.testing.assert_array_equal(
        r1["transition_state"].astype(str).to_numpy(),
        r2["transition_state"].astype(str).to_numpy(),
    )
    np.testing.assert_allclose(
        r1["transition_probability"].to_numpy(),
        r2["transition_probability"].to_numpy(),
        atol=1e-10,
    )
