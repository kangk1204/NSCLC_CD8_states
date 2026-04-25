from __future__ import annotations

from pathlib import Path

import pandas as pd

from nsclc_tf_switch.reporting import save_interaction_heatmap, save_patient_association_plot


def test_interaction_reporting_smoke(tmp_path: Path) -> None:
    interaction_df = pd.DataFrame(
        [
            {"sender_group": "myeloid", "ligand": "CD274", "receptor": "PDCD1", "delta_score": 1.0},
            {"sender_group": "cancer", "ligand": "PVR", "receptor": "TIGIT", "delta_score": 0.5},
            {"sender_group": "treg", "ligand": "TGFB1", "receptor": "TGFBR1", "delta_score": 0.4},
        ]
    )
    association_df = pd.DataFrame(
        [
            {"sender_group": "myeloid", "spearman_rho": 0.7},
            {"sender_group": "cancer", "spearman_rho": 0.2},
        ]
    )

    heatmap_path = save_interaction_heatmap(interaction_df, tmp_path)
    assoc_path = save_patient_association_plot(association_df, tmp_path)

    assert Path(heatmap_path).exists()
    assert Path(assoc_path).exists()
