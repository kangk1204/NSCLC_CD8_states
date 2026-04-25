from __future__ import annotations

from pathlib import Path

import requests

from nsclc_tf_switch.config import DATASETS, DatasetSpec


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        known = ", ".join(sorted(DATASETS))
        raise KeyError(f"Unknown dataset '{name}'. Known datasets: {known}") from exc


def download_file(url: str, destination: Path, chunk_size: int = 1 << 20) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing_size = destination.stat().st_size if destination.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
    mode = "ab" if existing_size else "wb"

    with requests.get(url, stream=True, timeout=60, headers=headers) as response:
        if existing_size and response.status_code == 200:
            destination.unlink(missing_ok=True)
            existing_size = 0
            mode = "wb"
        if response.status_code == 416:
            return destination
        response.raise_for_status()
        with destination.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    return destination


def download_named_dataset(name: str, output_dir: Path) -> Path:
    spec = get_dataset_spec(name)
    dataset_dir = output_dir / name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    destination = dataset_dir / spec.filename
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    return download_file(spec.url, destination)
