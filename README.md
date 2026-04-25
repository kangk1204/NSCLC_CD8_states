# NSCLC CD8 State Analysis

This repository contains the analysis code and reproducibility materials for a
public-data reanalysis of exhausted-like CD8 T-cell states in non-small cell
lung cancer (NSCLC).

The central result is intentionally restrained: exhausted-like CD8 states are
concentrated in specific T-cell states and are associated with
directionally recurrent checkpoint-oriented microenvironmental interactions.
The intrinsic TF layer remains discovery-rich but does not pass the strict
six-cohort validation rule, and TCGA LUAD/LUSC is used only as bounded bulk
gene-pair support for the recurrent interaction anchors.

## Research framing

The analysis is built around four questions:

1. Can graph representation learning separate activation-like and
   exhaustion-like T-cell states in the KU Leuven discovery cohort?
2. Which discovery TF signals survive a strict six-cohort external T-cell
   validation rule?
3. Which checkpoint-oriented ligand-receptor patterns recur across the
   discovery all-cell cohort plus five external all-cell cohorts?
4. Do TCGA LUAD/LUSC bulk RNA-seq data provide gene-pair support for the
   recurrent anchor interactions without resolving sender compartments?

## Public data sources

- KU Leuven NSCLC stromal/T-cell resource:
  `https://gbiomed.kuleuven.be/english/cme/research/laboratories/54213024/scRNAseq-NSCLC`
- NSCLC integrated metadata collection:
  `https://doi.org/10.6084/m9.figshare.c.6222221.v3`

The CLI knows the direct download endpoints for the KU Leuven T-cell loom and the Figshare metadata CSV.
It also supports the Figshare integrated raw-count matrix for cross-study validation.

## What the pipeline does

1. Downloads public data assets.
2. Loads a `.loom` matrix into `AnnData` using chunked sparse conversion.
3. Computes QC metrics and lightweight preprocessing without `scanpy`.
4. Builds a kNN graph from highly variable genes.
5. Trains a graph autoencoder with PyTorch Geometric to obtain a latent T-cell manifold.
6. Uses pseudo-label anchors from activation and exhaustion markers to infer a transition probability.
7. Estimates TF activity with DoRothEA + `decoupler`.
8. Ranks candidate TF switches and generates figures/tables.
9. Materializes independent validation cohorts from the integrated Figshare raw-count matrix.
10. Computes patient-aware ligand-receptor interaction scores from the KU all-cell loom.
11. Runs cross-cohort all-cell interaction recurrence summaries.
12. Adds TCGA LUAD/LUSC bulk correlation support for the predeclared recurrent
    anchor interaction genes.
13. Writes analysis outputs under ignored local result directories.

## Quick start

```bash
python3 -m pip install -e .
python3 -m nsclc_tf_switch.cli download --dataset ku_tcells
python3 -m nsclc_tf_switch.cli inspect-loom data/raw/ku_tcells/Thienpont_T-cell_v4_R_fixed.loom
python3 -m nsclc_tf_switch.cli analyze-loom \
  data/raw/ku_tcells/Thienpont_T-cell_v4_R_fixed.loom \
  --output-dir results/ku_tcells_run \
  --max-cells 4000 \
  --epochs 20

python3 -m nsclc_tf_switch.cli download --dataset figshare_rawcounts
python3 -m nsclc_tf_switch.cli run-validation-panel \
  --discovery-ranking results/full_ku_tcells/tf_switch_ranking.csv \
  --output-dir results/validation_panel

python3 -m nsclc_tf_switch.cli analyze-interactions \
  data/raw/ku_allcells/Thienpont_Tumors_52k_v4_R_fixed.loom \
  --output-dir results/interactions

python3 -m nsclc_tf_switch.cli run-interaction-validation \
  --discovery-interaction results/interactions/interaction_network.csv \
  --output-dir results/interaction_validation

```

`--max-cells` is useful for fast iteration. Omit it for a full run.

## Generated outputs

Generated analysis products are intentionally not tracked in git. Keep local
working outputs under ignored paths such as `results/`, `data/`, `paper/`,
`manuscript/`, `study_package/`, `submission/`, and `final_package/`.

The strongest interaction anchors are:

- `B/plasma CD274→PDCD1`
- `cancer CD274→PDCD1`
- `Treg LGALS9→HAVCR2`
- `B/plasma CD80→CTLA4`

These anchors preserve positive interaction delta and positive patient-level
Spearman ρ in all six cohorts, but they do not reach the strict discovery or
external q<0.1 significance thresholds. They should therefore be described as
directionally recurrent, not strictly validated.

## Snapshot provenance

Live analysis roots and manuscript/submission artifacts are intentionally absent
from this repository snapshot:

- `results/`
- `data/raw/`
- `paper/`
- `manuscript/`
- `study_package/`
- `submission/`
- `final_package/`

## Main outputs

- `results/.../prepared.h5ad`: preprocessed AnnData object
- `results/.../modeled.h5ad`: AnnData with graph latent and TF activity
- `results/.../tf_switch_ranking.csv`: ranked TF candidates
- `results/.../transition_embedding.png`: latent embedding colored by transition probability
- `results/.../top_tf_boxplot.png`: top TF activities across transition states
- `results/.../run_summary.json`: run metadata and summary statistics
- `results/validation_panel/validation_consensus_summary.csv`: cross-cohort TF validation summary
- `results/interactions/interaction_network.csv`: patient-aware ligand-receptor scores
- `results/interaction_validation/interaction_validation_summary.csv`: cross-cohort interaction validation summary
- `results/tcga_validation/tcga_anchor_edge_correlations.csv`: TCGA LUAD/LUSC
  bulk support for the anchor ligand-receptor genes

## Notes

- The environment already provides `anndata`, `decoupler`, `torch`, `torch-geometric`, and `scikit-learn`, so the implementation does not introduce extra dependencies beyond those already available.
- The current validation path is a smoke-tested synthetic workflow plus real-dataset download/inspection support. Running the full biological analysis on the public loom is expected to take longer than unit tests.
