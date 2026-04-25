from __future__ import annotations

from pathlib import Path

import pandas as pd


def summarize_integrated_metadata(path: str | Path) -> pd.DataFrame:
    metadata = pd.read_csv(path, low_memory=False)
    summary = (
        metadata.groupby(["Study", "Cell_Cluster_level1"], dropna=False)
        .size()
        .rename("cell_count")
        .reset_index()
        .sort_values(["Study", "cell_count"], ascending=[True, False])
    )
    return summary
