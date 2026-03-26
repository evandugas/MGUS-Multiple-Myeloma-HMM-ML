#!/usr/bin/env Rscript
# Process Affymetrix SNP 6.0 CEL files to extract log2 copy number ratios.
# Only needed if CNCHP files cannot be parsed by the Python pipeline.
#
# Usage: Rscript scripts/process_affymetrix_cel.R <input_dir> <output_dir>
#   input_dir:  directory containing .CEL.gz files
#   output_dir: where to write per-sample .txt files (chr, start, log2ratio)
#
# Requires: rawcopy (or aroma.affymetrix as fallback)
# Install: BiocManager::install("rawcopy")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Usage: Rscript process_affymetrix_cel.R <input_dir> <output_dir>")
}
input_dir <- args[1]
output_dir <- args[2]

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

if (!requireNamespace("rawcopy", quietly = TRUE)) {
    stop("rawcopy package not installed. Install with: BiocManager::install('rawcopy')")
}

library(rawcopy)

cel_files <- list.files(input_dir, pattern = "\\.CEL(\\.gz)?$",
                        full.names = TRUE, ignore.case = TRUE)
cat(sprintf("Found %d CEL files in %s\n", length(cel_files), input_dir))

for (cel_path in cel_files) {
    basename_cel <- basename(cel_path)
    gsm_id <- sub("_.*", "", basename_cel)
    out_file <- file.path(output_dir, paste0(gsm_id, "_log2ratio.txt"))

    if (file.exists(out_file)) {
        cat(sprintf("  %s: already processed\n", gsm_id))
        next
    }

    cat(sprintf("  Processing %s...\n", gsm_id))

    tryCatch({
        result <- rawcopy(cel_path, output = tempdir())

        # Extract copy number log2 ratios with coordinates
        cn_data <- data.frame(
            Chromosome = result$chromosome,
            Position = result$position,
            Log2Ratio = result$log2ratio
        )

        # Filter to autosomes
        cn_data <- cn_data[cn_data$Chromosome %in% 1:22, ]
        cn_data <- cn_data[!is.na(cn_data$Log2Ratio), ]

        write.table(cn_data, file = out_file, sep = "\t",
                    row.names = FALSE, quote = FALSE)
        cat(sprintf("  %s: %d probes written\n", gsm_id, nrow(cn_data)))
    }, error = function(e) {
        cat(sprintf("  %s: ERROR - %s\n", gsm_id, conditionMessage(e)))
    })
}

cat("Done.\n")
