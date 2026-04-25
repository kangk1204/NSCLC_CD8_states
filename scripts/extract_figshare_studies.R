args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 5) {
  stop("Usage: Rscript extract_figshare_studies.R <matrix_rds> <metadata_csv> <output_dir> <cell_level1> <study1> [study2 ...]")
}

matrix_rds <- args[[1]]
metadata_csv <- args[[2]]
output_dir <- args[[3]]
cell_level1 <- args[[4]]
studies <- args[5:length(args)]

suppressPackageStartupMessages(library(Matrix))

metadata <- read.csv(metadata_csv, check.names = FALSE)
cell_id_column <- if ("Unnamed: 0" %in% colnames(metadata)) {
  "Unnamed: 0"
} else {
  NULL
}
metadata$CellID <- if (is.null(cell_id_column)) {
  as.character(metadata[[1]])
} else {
  as.character(metadata[[cell_id_column]])
}
matrix_obj <- readRDS(matrix_rds)

if (inherits(matrix_obj, "dgTMatrix")) {
  matrix_obj <- as(matrix_obj, "dgCMatrix")
}

if (!(inherits(matrix_obj, "dgCMatrix") || is.matrix(matrix_obj))) {
  stop(paste("Unsupported matrix class:", paste(class(matrix_obj), collapse = ",")))
}

cell_ids <- metadata$CellID
col_overlap <- if (!is.null(colnames(matrix_obj))) sum(colnames(matrix_obj) %in% cell_ids) else 0
row_overlap <- if (!is.null(rownames(matrix_obj))) sum(rownames(matrix_obj) %in% cell_ids) else 0

if (row_overlap > col_overlap) {
  matrix_obj <- t(matrix_obj)
}

if (is.null(colnames(matrix_obj)) || is.null(rownames(matrix_obj))) {
  stop("Matrix must have rownames and colnames.")
}

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

for (study in studies) {
  keep <- metadata$Study == study
  if (!(cell_level1 %in% c("ALL", "*", "all"))) {
    keep <- keep & metadata$Cell_Cluster_level1 == cell_level1
  }
  subset_meta <- metadata[keep, , drop = FALSE]
  common_cells <- intersect(subset_meta$CellID, colnames(matrix_obj))
  if (length(common_cells) == 0) {
    stop(paste("No overlapping cells found for study:", study))
  }

  subset_meta <- subset_meta[match(common_cells, subset_meta$CellID), , drop = FALSE]
  subset_matrix <- matrix_obj[, common_cells, drop = FALSE]
  subset_dir <- file.path(output_dir, study)
  dir.create(subset_dir, recursive = TRUE, showWarnings = FALSE)

  writeMM(subset_matrix, file = file.path(subset_dir, "matrix.mtx"))
  write.table(
    rownames(subset_matrix),
    file = file.path(subset_dir, "features.tsv"),
    quote = FALSE,
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE
  )
  write.csv(subset_meta, file = file.path(subset_dir, "obs.csv"), row.names = FALSE)
}
