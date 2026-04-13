#!/usr/bin/env Rscript
# CBS segmentation on probe-level aCGH data using DNAcopy.
#
# Input:  TSV file with columns: sample_id, chr, start, LogRatio
#         (all samples concatenated, one probe per row)
# Output: TSV file with CBS segments: sample_id, chr, start, end, n_probes, seg_mean
#
# Usage: Rscript pipeline/10_cbs_segment.R <input.tsv> <output.tsv>

suppressPackageStartupMessages(library(DNAcopy))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Usage: Rscript 10_cbs_segment.R <input_probes.tsv> <output_segments.tsv>")
}
input_file <- args[1]
output_file <- args[2]

cat("Reading probe data...\n")
probes <- read.delim(input_file, stringsAsFactors = FALSE)
cat(sprintf("  %d probes, %d samples\n", nrow(probes), length(unique(probes$sample_id))))

# DNAcopy wants chromosome as integer
probes$chr_num <- as.integer(sub("chr", "", probes$chr))
probes <- probes[!is.na(probes$chr_num), ]

samples <- unique(probes$sample_id)
all_segments <- list()

cat("Running CBS segmentation...\n")
for (i in seq_along(samples)) {
    sid <- samples[i]
    sp <- probes[probes$sample_id == sid, ]
    sp <- sp[order(sp$chr_num, sp$start), ]

    # Create CNA object
    cna_obj <- CNA(
        genomdat = sp$LogRatio,
        chrom = sp$chr_num,
        maploc = sp$start,
        data.type = "logratio",
        sampleid = sid
    )

    # Smooth outliers
    cna_smooth <- smooth.CNA(cna_obj)

    # CBS segmentation
    cbs_result <- segment(cna_smooth, verbose = 0, min.width = 3)

    # Extract segments
    segs <- cbs_result$output
    segs$sample_id <- sid
    segs$chr <- paste0("chr", segs$chrom)
    all_segments[[i]] <- segs[, c("sample_id", "chr", "loc.start", "loc.end",
                                   "num.mark", "seg.mean")]

    if (i %% 20 == 0) {
        cat(sprintf("  %d/%d samples done\n", i, length(samples)))
    }
}

cat(sprintf("  %d/%d samples done\n", length(samples), length(samples)))

result <- do.call(rbind, all_segments)
colnames(result) <- c("sample_id", "chr", "start", "end", "n_probes", "seg_mean")

write.table(result, file = output_file, sep = "\t", row.names = FALSE, quote = FALSE)
cat(sprintf("Wrote %d segments to %s\n", nrow(result), output_file))
