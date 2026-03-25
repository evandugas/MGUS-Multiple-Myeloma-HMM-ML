###############################################################################
# Phase 1: Data Loading, EDA, Cleaning, and Export
# Loads Agilent CGH array data from GSE77975, performs QC, and exports
# cleaned data for HMM/ML pipeline in Python.
###############################################################################

.libPaths(Sys.getenv("R_LIBS_USER"))
library(data.table)
library(ggplot2)
library(parallel)

base_dir <- file.path("C:", "Users", "dugas",
                       "OneDrive - Boston University", "MA770",
                       "MM_MGUS_Project")
n_cores <- detectCores() - 2  # leave 2 cores free

# ---- 1. Load and parse all sample files ------------------------------------

parse_agilent_fe <- function(filepath, sample_id, label) {
  # fread skip= finds the header row directly — no double-read
  cols_select <- c("ControlType", "ProbeName", "SystematicName",
                   "LogRatio", "LogRatioError")
  dt <- fread(
    file = filepath,
    skip = "FEATURES",
    sep = "\t",
    select = cols_select,
    colClasses = list(
      integer = "ControlType",
      character = c("ProbeName", "SystematicName"),
      numeric = c("LogRatio", "LogRatioError")
    )
  )
  # Drop the FEATURES header row itself if fread included it as data
  dt <- dt[!is.na(LogRatio)]
  dt[, sample_id := sample_id]
  dt[, label := label]
  dt
}

# Build file list with labels
mgus_files <- list.files(file.path(base_dir, "data", "raw", "mgus"),
                         pattern = "\\.gz$", full.names = TRUE)
mm_files   <- list.files(file.path(base_dir, "data", "raw", "mm"),
                         pattern = "\\.gz$", full.names = TRUE)

file_info <- data.table(
  filepath = c(mgus_files, mm_files),
  label    = c(rep("MGUS", length(mgus_files)), rep("MM", length(mm_files)))
)
file_info[, sample_id := sub("_.*", "", basename(filepath))]

cat(sprintf("Found %d MGUS and %d MM files (%d total)\n",
            length(mgus_files), length(mm_files), nrow(file_info)))

# Parallel load across cores
cat(sprintf("Loading samples in parallel (%d cores)...\n", n_cores))
t0 <- proc.time()

cl <- makeCluster(n_cores)
clusterEvalQ(cl, { .libPaths(Sys.getenv("R_LIBS_USER")); library(data.table) })
clusterExport(cl, c("parse_agilent_fe", "file_info"))

all_data <- rbindlist(
  parLapply(cl, seq_len(nrow(file_info)), function(i) {
    parse_agilent_fe(file_info$filepath[i],
                     file_info$sample_id[i],
                     file_info$label[i])
  }),
  fill = TRUE
)
stopCluster(cl)

elapsed <- (proc.time() - t0)["elapsed"]
cat(sprintf("Loaded %s rows across %d samples in %.1f seconds\n",
            format(nrow(all_data), big.mark = ","), nrow(file_info), elapsed))

# ---- 2. Filter control probes and parse genomic coordinates ----------------

cat(sprintf("Control probes: %s  |  Non-control: %s\n",
            format(sum(all_data$ControlType != 0), big.mark = ","),
            format(sum(all_data$ControlType == 0), big.mark = ",")))

all_data <- all_data[ControlType == 0]

# Parse SystematicName (e.g., "chr12:24273514-24273573")
all_data[, c("chr", "pos_range") := tstrsplit(SystematicName, ":", fixed = TRUE)]
all_data[, c("start", "end") := tstrsplit(pos_range, "-", fixed = TRUE, type.convert = TRUE)]
all_data[, pos_range := NULL]
all_data[, start := as.integer(start)]
all_data[, end := as.integer(end)]

# Check for unparseable coordinates
n_bad_coord <- sum(is.na(all_data$chr) | is.na(all_data$start))
cat(sprintf("Probes with unparseable coordinates: %d\n", n_bad_coord))
if (n_bad_coord > 0) all_data <- all_data[!is.na(chr) & !is.na(start)]

# Summary of chromosomes
cat("\nProbes per chromosome (first sample):\n")
print(all_data[sample_id == all_data$sample_id[1], .N, by = chr][order(chr)])

# ---- 3. EDA ---------------------------------------------------------------

plot_dir <- file.path(base_dir, "output", "plots")

# 3a. LogRatio distribution per sample (boxplots)
sample_summary <- all_data[, .(
  median_lr  = median(LogRatio, na.rm = TRUE),
  mean_lr    = mean(LogRatio, na.rm = TRUE),
  sd_lr      = sd(LogRatio, na.rm = TRUE),
  iqr_lr     = IQR(LogRatio, na.rm = TRUE),
  n_probes   = .N,
  median_err = median(LogRatioError, na.rm = TRUE)
), by = .(sample_id, label)]

p1 <- ggplot(sample_summary, aes(x = reorder(sample_id, median_lr),
                                  y = median_lr, fill = label)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "Median LogRatio per Sample", x = "Sample", y = "Median LogRatio") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 3))
ggsave(file.path(plot_dir, "01_median_logratio_per_sample.png"),
       p1, width = 10, height = 14, dpi = 150)

# 3b. LogRatio SD per sample (noise level)
p2 <- ggplot(sample_summary, aes(x = reorder(sample_id, sd_lr),
                                  y = sd_lr, fill = label)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  labs(title = "LogRatio SD per Sample (noise)", x = "Sample", y = "SD LogRatio") +
  theme_minimal() +
  theme(axis.text.y = element_text(size = 3))
ggsave(file.path(plot_dir, "02_sd_logratio_per_sample.png"),
       p2, width = 10, height = 14, dpi = 150)

# 3c. Overall LogRatio density by label
set.seed(42)
subsample <- all_data[sample(.N, min(.N, 1e6))]
p3 <- ggplot(subsample, aes(x = LogRatio, color = label)) +
  geom_density(linewidth = 0.8) +
  xlim(-2, 2) +
  labs(title = "LogRatio Density: MGUS vs MM", x = "LogRatio", y = "Density") +
  theme_minimal()
ggsave(file.path(plot_dir, "03_logratio_density_by_label.png"),
       p3, width = 8, height = 5, dpi = 150)

# 3d. Batch effect check: extract slide ID from filename
file_info[, slide_id := sub(".*_(\\d{12})_.*", "\\1", basename(filepath))]
sample_summary <- merge(sample_summary, file_info[, .(sample_id, slide_id)],
                        by = "sample_id")

p4 <- ggplot(sample_summary, aes(x = slide_id, y = median_lr, color = label)) +
  geom_jitter(width = 0.2, size = 2, alpha = 0.7) +
  labs(title = "Median LogRatio by Array Slide (batch check)",
       x = "Slide ID", y = "Median LogRatio") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, size = 6))
ggsave(file.path(plot_dir, "04_batch_effect_by_slide.png"),
       p4, width = 12, height = 6, dpi = 150)

# 3e. Probe count per sample (should be uniform)
p5 <- ggplot(sample_summary, aes(x = n_probes, fill = label)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(title = "Probe Count per Sample", x = "Number of Probes", y = "Count") +
  theme_minimal()
ggsave(file.path(plot_dir, "05_probe_count_histogram.png"),
       p5, width = 8, height = 5, dpi = 150)

cat("\nSample summary statistics:\n")
print(summary(sample_summary[, .(median_lr, sd_lr, n_probes, median_err)]))

# Flag outlier samples (SD > 3x median SD)
median_sd <- median(sample_summary$sd_lr)
outliers <- sample_summary[sd_lr > 3 * median_sd]
if (nrow(outliers) > 0) {
  cat(sprintf("\nWARNING: %d potential outlier samples (high noise):\n",
              nrow(outliers)))
  print(outliers[, .(sample_id, label, sd_lr)])
} else {
  cat("\nNo outlier samples detected by noise threshold.\n")
}

# ---- 4. Cleaning -----------------------------------------------------------

cat("\n--- Cleaning ---\n")

# 4a. Drop sex chromosomes
n_before <- nrow(all_data)
all_data <- all_data[!chr %in% c("chrX", "chrY")]
cat(sprintf("Dropped sex chromosomes: %s probes removed\n",
            format(n_before - nrow(all_data), big.mark = ",")))

# 4b. Remove probes with high error
error_threshold <- quantile(all_data$LogRatioError, 0.99, na.rm = TRUE)
n_before <- nrow(all_data)
all_data <- all_data[LogRatioError <= error_threshold]
cat(sprintf("Removed probes with LogRatioError > %.4f (99th pctl): %s probes\n",
            error_threshold, format(n_before - nrow(all_data), big.mark = ",")))

# 4c. Median-center LogRatio per sample
all_data[, LogRatio_raw := LogRatio]
all_data[, LogRatio := LogRatio - median(LogRatio, na.rm = TRUE), by = sample_id]

cat(sprintf("\nAfter cleaning: %s probes across %d samples\n",
            format(nrow(all_data), big.mark = ","),
            uniqueN(all_data$sample_id)))

# Verify centering
centering_check <- all_data[, .(median_lr = median(LogRatio)), by = sample_id]
cat(sprintf("Post-centering median LogRatio range: [%.6f, %.6f]\n",
            min(centering_check$median_lr), max(centering_check$median_lr)))

# ---- 5. Export for Python --------------------------------------------------

cat("\n--- Exporting ---\n")
output_dir <- file.path(base_dir, "data", "cleaned")
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# 5a. Sample metadata
meta <- unique(all_data[, .(sample_id, label)])
meta <- merge(meta, file_info[, .(sample_id, slide_id)], by = "sample_id")
fwrite(meta, file.path(output_dir, "sample_metadata.csv"))
cat(sprintf("Saved sample_metadata.csv (%d samples)\n", nrow(meta)))

# 5b. Probe-level data (long format, sorted by chr + position)
# Chromosome sort order
chr_order <- paste0("chr", c(1:22))
all_data[, chr_num := match(chr, chr_order)]
all_data <- all_data[!is.na(chr_num)]  # drop any non-autosomal that slipped through
setorder(all_data, sample_id, chr_num, start)

export_cols <- c("sample_id", "label", "chr", "start", "end",
                 "ProbeName", "LogRatio", "LogRatioError")
fwrite(all_data[, ..export_cols],
       file.path(output_dir, "probe_data_cleaned.csv.gz"),
       compress = "gzip")
cat(sprintf("Saved probe_data_cleaned.csv.gz (%s rows)\n",
            format(nrow(all_data), big.mark = ",")))

# 5c. Probe map (unique probe positions, shared across samples)
probe_map <- unique(all_data[, .(ProbeName, chr, chr_num, start, end)])
setorder(probe_map, chr_num, start)
fwrite(probe_map, file.path(output_dir, "probe_map.csv"))
cat(sprintf("Saved probe_map.csv (%s unique probes)\n",
            format(nrow(probe_map), big.mark = ",")))

cat("\nPhase 1 complete.\n")
cat(sprintf("Plots saved to: %s\n", plot_dir))
cat(sprintf("Data saved to:  %s\n", output_dir))
