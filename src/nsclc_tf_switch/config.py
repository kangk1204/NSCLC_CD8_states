from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    url: str
    filename: str
    citation: str
    description: str


@dataclass(frozen=True)
class LigandReceptorPair:
    ligand: str
    receptor: str
    pathway: str


DATASETS: dict[str, DatasetSpec] = {
    "ku_tcells": DatasetSpec(
        name="ku_tcells",
        url="https://gbiomed.kuleuven.be/english/cme/research/laboratories/54213024/scrnaseq_tutorial/t-cells",
        filename="Thienpont_T-cell_v4_R_fixed.loom",
        citation=(
            "KU Leuven NSCLC stromal resource, T-cell loom file "
            "(accessed via the public download endpoint)."
        ),
        description="KU Leuven NSCLC T-cell single-cell RNA-seq loom file.",
    ),
    "ku_allcells": DatasetSpec(
        name="ku_allcells",
        url="https://gbiomed.kuleuven.be/english/cme/research/laboratories/54213024/scrnaseq_tutorial/all-cells",
        filename="Thienpont_Tumors_52k_v4_R_fixed.loom",
        citation="KU Leuven NSCLC stromal resource, all-cell loom file.",
        description="KU Leuven NSCLC all-cell single-cell RNA-seq loom file.",
    ),
    "figshare_metadata": DatasetSpec(
        name="figshare_metadata",
        url="https://ndownloader.figshare.com/files/39537250",
        filename="metadata.csv",
        citation=(
            "Prazanowska K, Lim SB. An integrated single-cell transcriptomic dataset for "
            "non-small cell lung cancer. figshare. https://doi.org/10.6084/m9.figshare.c.6222221.v3"
        ),
        description="Integrated NSCLC metadata CSV from the linked Figshare collection.",
    ),
    "figshare_rawcounts": DatasetSpec(
        name="figshare_rawcounts",
        url="https://ndownloader.figshare.com/files/39537094",
        filename="RNA_rawcounts_matrix.rds",
        citation=(
            "Prazanowska K, Lim SB. RNA raw-count matrix from the integrated NSCLC "
            "Figshare collection. https://doi.org/10.6084/m9.figshare.c.6222221.v3"
        ),
        description="Integrated NSCLC raw-count matrix in RDS format.",
    ),
}


ACTIVATION_MARKERS = [
    "IL7R",
    "LTB",
    "MALAT1",
    "GZMK",
    "CXCR3",
    "IFNG",
]

EXHAUSTION_MARKERS = [
    "PDCD1",
    "CTLA4",
    "LAG3",
    "TIGIT",
    "HAVCR2",
    "TOX",
]

STATE_LABELS = {
    "activated": "activated_anchor",
    "boundary": "transition_boundary",
    "exhausted": "exhausted_anchor",
}

DEFAULT_VALIDATION_STUDIES = ("GSE131907", "GSE136246")
DEFAULT_INTERACTION_VALIDATION_STUDIES = ("GSE131907", "GSE127465")

PAPER_TF_PANEL = (
    "STAT2",
    "IRF9",
    "IRF1",
    "HSF1",
    "RFX5",
    "STAT4",
    "NFATC1",
    "FOXO1",
    "AHR",
    "STAT1",
    "STAT5B",
    "DUX4",
)

INTERACTION_PANEL = (
    LigandReceptorPair("CD274", "PDCD1", "checkpoint"),
    LigandReceptorPair("PDCD1LG2", "PDCD1", "checkpoint"),
    LigandReceptorPair("CD80", "CTLA4", "checkpoint"),
    LigandReceptorPair("CD86", "CTLA4", "checkpoint"),
    LigandReceptorPair("PVR", "TIGIT", "checkpoint"),
    LigandReceptorPair("NECTIN2", "TIGIT", "checkpoint"),
    LigandReceptorPair("LGALS9", "HAVCR2", "checkpoint"),
    LigandReceptorPair("CXCL9", "CXCR3", "chemokine"),
    LigandReceptorPair("CXCL10", "CXCR3", "chemokine"),
    LigandReceptorPair("CXCL11", "CXCR3", "chemokine"),
    LigandReceptorPair("CCL19", "CCR7", "chemokine"),
    LigandReceptorPair("CCL21", "CCR7", "chemokine"),
    LigandReceptorPair("CXCL12", "CXCR4", "chemokine"),
    LigandReceptorPair("TGFB1", "TGFBR1", "tgfb"),
    LigandReceptorPair("TGFB1", "TGFBR2", "tgfb"),
    LigandReceptorPair("IL10", "IL10RA", "cytokine"),
    LigandReceptorPair("IL10", "IL10RB", "cytokine"),
    LigandReceptorPair("IFNG", "IFNGR1", "interferon"),
    LigandReceptorPair("IFNG", "IFNGR2", "interferon"),
    LigandReceptorPair("IFNA1", "IFNAR1", "interferon"),
    LigandReceptorPair("IFNA1", "IFNAR2", "interferon"),
    LigandReceptorPair("IFNB1", "IFNAR1", "interferon"),
    LigandReceptorPair("IFNB1", "IFNAR2", "interferon"),
    LigandReceptorPair("TNF", "TNFRSF1A", "tnf"),
    LigandReceptorPair("TNF", "TNFRSF1B", "tnf"),
    LigandReceptorPair("ICOSLG", "ICOS", "costimulatory"),
    LigandReceptorPair("TNFSF9", "TNFRSF9", "costimulatory"),
    LigandReceptorPair("FASLG", "FAS", "death_receptor"),
    LigandReceptorPair("TNFSF10", "TNFRSF10B", "death_receptor"),
)
