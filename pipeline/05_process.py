#!/usr/bin/env python3
"""Unified data processing pipeline for all datasets.

Processes each dataset independently through:
  1. Parse raw files (Agilent FE or Affymetrix processed)
  2. Clean (filter controls, drop sex chr, remove high-error probes, median-center)
  3. Bin probes into fixed 1Mb genomic bins for cross-platform normalization
  4. Run HMM segmentation (with pooled transition matrix) on binned data
  5. Extract arm-level CNA features + derived features
  6. Export per-dataset results
Then merges all arm-level features into one master matrix.

v3: Genomic bin normalization for multi-platform harmonization.
    Supports Agilent aCGH and Affymetrix SNP 6.0 data.
"""

import os
import sys
import gc
import warnings
import csv
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))
import importlib
_genomic_bins = importlib.import_module("02_genomic_bins")
_parsers = importlib.import_module("03_parsers")
_hmm_core = importlib.import_module("04_hmm_core")

fit_hmm_chromosome = _hmm_core.fit_hmm_chromosome
postprocess_states = _hmm_core.postprocess_states
estimate_pooled_transmat = _hmm_core.estimate_pooled_transmat
DEFAULT_MIN_CNA_PROBES = _hmm_core.DEFAULT_MIN_CNA_PROBES

CHROMOSOMES = _genomic_bins.CHROMOSOMES
CENTROMERES = _genomic_bins.CENTROMERES
N_BINS = _genomic_bins.N_BINS
BIN_TABLE = _genomic_bins.BIN_TABLE
compute_bin_values = _genomic_bins.compute_bin_values
binned_to_chrom_arrays = _genomic_bins.binned_to_chrom_arrays
bins_to_arm_mapping = _genomic_bins.bins_to_arm_mapping

parse_agilent_file = _parsers.parse_agilent_file
detect_and_parse_affymetrix = _parsers.detect_and_parse_affymetrix

N_WORKERS = 6

# Dataset registry: name -> (platform_type, parser_function)
DATASET_CONFIG = {
    "GSE77975": {"platform": "agilent", "parser": parse_agilent_file},
    "GSE33685": {"platform": "agilent", "parser": parse_agilent_file},
    "GSE26849": {"platform": "agilent", "parser": parse_agilent_file},
    "GSE44745": {"platform": "agilent", "parser": parse_agilent_file},
    "GSE29023": {"platform": "agilent", "parser": parse_agilent_file},
    "GSE31339": {"platform": "affymetrix", "parser": detect_and_parse_affymetrix},
}


# ============================================================================
# PROCESSING PIPELINE
# ============================================================================

def clean_probe_data(rows_list, has_error_col=True):
    """Clean parsed probe data: remove high-error probes, median-center."""
    df = pd.DataFrame(rows_list,
                      columns=["sample_id", "label", "chr", "start", "end",
                               "LogRatio", "LogRatioError"])

    if has_error_col and df["LogRatioError"].max() > 0:
        err_thresh = df["LogRatioError"].quantile(0.99)
        df = df[df["LogRatioError"] <= err_thresh]

    # Median-center per sample
    medians = df.groupby("sample_id")["LogRatio"].transform("median")
    df["LogRatio"] = df["LogRatio"] - medians

    return df


def bin_probe_data(probe_df):
    """Bin probe data into 1Mb genomic bins per sample.

    Args:
        probe_df: DataFrame with columns [sample_id, label, chr, start, LogRatio]

    Returns:
        DataFrame with columns [sample_id, bin_idx, log_ratio] where each
        row is one bin for one sample. Bins with no probes are omitted.
    """
    rows = []
    arm_map = bins_to_arm_mapping()

    for sample_id, sdf in probe_df.groupby("sample_id"):
        bin_values = compute_bin_values(sdf)

        # Per-sample centering on binned values (re-center after binning)
        valid = ~np.isnan(bin_values)
        if valid.sum() > 0:
            bin_values[valid] -= np.median(bin_values[valid])

        for i in range(N_BINS):
            if not np.isnan(bin_values[i]):
                rows.append((sample_id, i, float(bin_values[i])))

    return pd.DataFrame(rows, columns=["sample_id", "bin_idx", "log_ratio"])


def run_hmm_on_binned(binned_df):
    """Run HMM on binned data with pooled transition matrices.

    Two-pass approach per sample:
      Pass 1: Fit each chromosome independently -> pooled transition matrix
      Pass 2: Re-fit each chromosome using the pooled transition matrix
    """
    executor = ProcessPoolExecutor(max_workers=N_WORKERS)
    all_states = []

    for sample_id, sample_df in binned_df.groupby("sample_id"):
        # Convert bin indices to per-chromosome arrays
        sample_bins = sample_df.set_index("bin_idx")["log_ratio"]
        bin_values = np.full(N_BINS, np.nan)
        for idx, val in sample_bins.items():
            bin_values[idx] = val

        chrom_data = binned_to_chrom_arrays(bin_values)

        if not chrom_data:
            continue

        # Pass 1: Estimate pooled transition matrix
        all_lr = {chrom: data[0] for chrom, data in chrom_data.items()
                  if len(data[0]) >= 10}
        if not all_lr:
            continue
        pooled_transmat = estimate_pooled_transmat(all_lr)

        # Pass 2: Re-fit with pooled transition matrix
        futures = {}
        for chrom, (lr, midpoints, indices) in chrom_data.items():
            if len(lr) < 10:
                continue
            # Use midpoints as positions, create dummy ends
            ends = midpoints + 500_000  # half bin size
            futures[chrom] = executor.submit(
                fit_hmm_chromosome, (chrom, lr, midpoints, ends), pooled_transmat)

        for chrom, fut in futures.items():
            result = fut.result()
            if result is None:
                continue
            _, states, _, _ = result
            states = postprocess_states(states)
            lr, midpoints, indices = chrom_data[chrom]
            for i, state in enumerate(states):
                all_states.append((sample_id, chrom, int(midpoints[i]),
                                   float(lr[i]), int(state), int(indices[i])))

    executor.shutdown()
    return pd.DataFrame(all_states,
                        columns=["sample_id", "chr", "start", "log_ratio",
                                 "state", "bin_idx"])


def compute_arm_features(probe_states_df, labels_dict):
    """Compute arm-level CNA features from HMM states.

    Returns both basic arm fractions and derived features:
      - del_*/amp_* fractions per arm (88 features)
      - cna_burden: fraction of all bins in any CNA state
      - n_segments, n_del_segments, n_amp_segments
      - seg_count_*: CNA segments per arm
      - seg_mean_del_*/seg_mean_amp_*: mean log-ratio in CNA segments per arm
    """
    df = probe_states_df.copy()

    # Assign arms
    df["arm"] = "q"
    for chrom, centro in CENTROMERES.items():
        mask = (df["chr"] == chrom) & (df["start"] < centro)
        df.loc[mask, "arm"] = "p"
    df["arm_id"] = df["chr"] + df["arm"]

    # Basic arm fractions
    total = df.groupby(["sample_id", "arm_id"]).size().reset_index(name="n_total")
    n_del = df[df["state"] == 0].groupby(
        ["sample_id", "arm_id"]).size().reset_index(name="n_del")
    n_amp = df[df["state"] == 2].groupby(
        ["sample_id", "arm_id"]).size().reset_index(name="n_amp")

    feat = total.merge(n_del, on=["sample_id", "arm_id"], how="left")
    feat = feat.merge(n_amp, on=["sample_id", "arm_id"], how="left")
    feat["n_del"] = feat["n_del"].fillna(0).astype(int)
    feat["n_amp"] = feat["n_amp"].fillna(0).astype(int)
    feat["frac_del"] = feat["n_del"] / feat["n_total"]
    feat["frac_amp"] = feat["n_amp"] / feat["n_total"]

    del_frac = feat.pivot_table(index="sample_id", columns="arm_id",
                                 values="frac_del", fill_value=0)
    amp_frac = feat.pivot_table(index="sample_id", columns="arm_id",
                                 values="frac_amp", fill_value=0)
    del_frac.columns = [f"del_{c}" for c in del_frac.columns]
    amp_frac.columns = [f"amp_{c}" for c in amp_frac.columns]

    result = pd.concat([del_frac, amp_frac], axis=1)

    # CNA burden
    sample_totals = df.groupby("sample_id").size()
    sample_cna = df[df["state"] != 1].groupby("sample_id").size()
    cna_burden = (sample_cna / sample_totals).fillna(0)
    result["cna_burden"] = result.index.map(cna_burden).fillna(0)

    # Segment counts
    seg_counts = defaultdict(lambda: {"n_segments": 0, "n_del_segments": 0,
                                       "n_amp_segments": 0})
    arm_seg_counts = defaultdict(lambda: defaultdict(int))
    arm_seg_means_del = defaultdict(lambda: defaultdict(list))
    arm_seg_means_amp = defaultdict(lambda: defaultdict(list))

    for (sample_id, chrom), group in df.groupby(["sample_id", "chr"]):
        group = group.sort_values("start")
        states = group["state"].values
        log_ratios = group["log_ratio"].values
        arms = group["arm_id"].values

        i = 0
        while i < len(states):
            if states[i] != 1:
                j = i
                while j < len(states) and states[j] == states[i]:
                    j += 1
                seg_counts[sample_id]["n_segments"] += 1
                seg_lr = log_ratios[i:j]
                seg_arm = arms[i]

                if states[i] == 0:
                    seg_counts[sample_id]["n_del_segments"] += 1
                    arm_seg_counts[sample_id][seg_arm] += 1
                    arm_seg_means_del[sample_id][seg_arm].append(float(np.mean(seg_lr)))
                elif states[i] == 2:
                    seg_counts[sample_id]["n_amp_segments"] += 1
                    arm_seg_counts[sample_id][seg_arm] += 1
                    arm_seg_means_amp[sample_id][seg_arm].append(float(np.mean(seg_lr)))
                i = j
            else:
                i += 1

    all_samples = result.index
    result["n_segments"] = [seg_counts[s]["n_segments"] for s in all_samples]
    result["n_del_segments"] = [seg_counts[s]["n_del_segments"] for s in all_samples]
    result["n_amp_segments"] = [seg_counts[s]["n_amp_segments"] for s in all_samples]

    all_arms = sorted(set(df["arm_id"]))
    for arm in all_arms:
        result[f"seg_count_{arm}"] = [arm_seg_counts[s].get(arm, 0) for s in all_samples]

    for arm in all_arms:
        del_means = []
        amp_means = []
        for s in all_samples:
            dm = arm_seg_means_del[s].get(arm, [])
            am = arm_seg_means_amp[s].get(arm, [])
            del_means.append(float(np.mean(dm)) if dm else 0.0)
            amp_means.append(float(np.mean(am)) if am else 0.0)
        result[f"seg_mean_del_{arm}"] = del_means
        result[f"seg_mean_amp_{arm}"] = amp_means

    result.insert(0, "label", result.index.map(labels_dict))

    return result


def process_dataset(dataset_name, data_dir):
    """Process a dataset: parse -> clean -> bin -> HMM -> arm features."""
    config = DATASET_CONFIG[dataset_name]
    parser = config["parser"]
    platform = config["platform"]

    print(f"\n{'='*60}")
    print(f"  Processing {dataset_name} ({platform})")
    print(f"{'='*60}")

    raw_dir = os.path.join(data_dir, "raw")
    labels_csv = os.path.join(data_dir, "sample_labels.csv")

    if not os.path.exists(labels_csv):
        print(f"  WARNING: {labels_csv} not found, skipping")
        return None

    # Read labels
    labels_dict = {}
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            labels_dict[row["sample_id"]] = row["label"]

    # Remap mm_excluded to MM
    labels_dict = {k: ("MM" if v == "mm_excluded" else v)
                   for k, v in labels_dict.items()}

    # Collect files from mgus/ and mm/
    all_rows = []
    for subdir, label in [("mgus", "MGUS"), ("mm", "MM")]:
        dir_path = os.path.join(raw_dir, subdir)
        if not os.path.exists(dir_path):
            continue
        files = sorted([f for f in os.listdir(dir_path)
                        if f.endswith(".gz") or f.endswith(".txt") or f.endswith(".CEL")])
        print(f"  {subdir}/: {len(files)} files")
        for filename in files:
            gsm_id = filename.split("_")[0].split(".")[0]
            filepath = os.path.join(dir_path, filename)
            rows = parser(filepath, gsm_id, label)
            if rows:
                all_rows.extend(rows)

    if not all_rows:
        print(f"  WARNING: No data parsed for {dataset_name}")
        return None

    print(f"  Parsed {len(all_rows):,} probe rows")

    # Clean
    has_error = platform == "agilent"
    probe_df = clean_probe_data(all_rows, has_error_col=has_error)
    del all_rows
    gc.collect()
    n_samples = probe_df["sample_id"].nunique()
    print(f"  After cleaning: {len(probe_df):,} rows, {n_samples} samples")

    # Bin probes into 1Mb genomic bins
    print(f"  Binning into 1Mb bins ({N_BINS} bins per sample)...")
    binned_df = bin_probe_data(probe_df)
    bins_per_sample = binned_df.groupby("sample_id").size()
    mean_bins = bins_per_sample.mean()
    min_bins = bins_per_sample.min()
    print(f"  Binned: avg {mean_bins:.0f} bins/sample (min {min_bins}), "
          f"{binned_df['sample_id'].nunique()} samples")

    del probe_df
    gc.collect()

    # HMM on binned data
    print(f"  Running HMM on binned data...")
    probe_states = run_hmm_on_binned(binned_df)
    print(f"  HMM complete: {len(probe_states):,} bin states")

    # Arm features
    arm_features = compute_arm_features(probe_states, labels_dict)
    n_basic = sum(1 for c in arm_features.columns if c.startswith(("del_", "amp_"))
                  and not c.startswith(("seg_mean_del", "seg_mean_amp")))
    n_derived = arm_features.shape[1] - n_basic - 1
    print(f"  Features: {n_basic} arm fractions + {n_derived} derived = "
          f"{arm_features.shape[1] - 1} total")

    # Save per-dataset
    out_dir = os.path.join(data_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)
    arm_features.to_csv(os.path.join(out_dir, "arm_features.csv"))

    # Also save bin-level data for ComBat
    bin_matrix = binned_df.pivot_table(index="sample_id", columns="bin_idx",
                                        values="log_ratio", fill_value=0)
    bin_matrix.to_csv(os.path.join(out_dir, "bin_features.csv"))
    print(f"  Saved to {out_dir}/")

    return arm_features, bin_matrix


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Multi-Platform Processing Pipeline (v3 — binned HMM)")
    print("=" * 60)

    all_arm_features = []
    all_bin_features = []

    for ds_name in DATASET_CONFIG:
        ds_dir = os.path.join(DATA_DIR, ds_name)
        if not os.path.exists(ds_dir):
            print(f"\n  Skipping {ds_name} (directory not found)")
            continue
        result = process_dataset(ds_name, ds_dir)
        if result is not None:
            arm_feat, bin_feat = result
            all_arm_features.append((ds_name, arm_feat))
            all_bin_features.append((ds_name, bin_feat))

    # --- Merge all features ---
    if all_arm_features:
        print(f"\n{'='*60}")
        print(f"  Merging {len(all_arm_features)} datasets")
        print(f"{'='*60}")

        # Merge arm-level features
        merged_parts = []
        for ds_name, df in all_arm_features:
            df = df.copy()
            df.insert(1, "dataset", ds_name)
            merged_parts.append(df)
            mgus = (df["label"] == "MGUS").sum()
            mm = (df["label"] == "MM").sum()
            print(f"  {ds_name}: {mgus} MGUS + {mm} MM = {len(df)}")

        merged = pd.concat(merged_parts, ignore_index=False)
        merged = merged.fillna(0)

        feat_dir = os.path.join(OUTPUT_DIR, "features")
        os.makedirs(feat_dir, exist_ok=True)
        merged.to_csv(os.path.join(feat_dir, "feature_matrix_arm_all.csv"))

        total_mgus = (merged["label"] == "MGUS").sum()
        total_mm = (merged["label"] == "MM").sum()
        n_feat = merged.shape[1] - 2
        print(f"\n  MERGED: {total_mgus} MGUS + {total_mm} MM = {len(merged)} samples")
        print(f"  Features: {n_feat} total")
        print(f"  Saved to {feat_dir}/feature_matrix_arm_all.csv")

        # Merge bin-level features
        bin_parts = []
        for ds_name, df in all_bin_features:
            df = df.copy()
            df.insert(0, "dataset", ds_name)
            # Add label from arm features
            arm_labels = all_arm_features[[n for n, _ in all_arm_features].index(ds_name)][1]["label"]
            df.insert(0, "label", df.index.map(arm_labels))
            bin_parts.append(df)

        merged_bins = pd.concat(bin_parts, ignore_index=False)
        merged_bins = merged_bins.fillna(0)
        merged_bins.to_csv(os.path.join(feat_dir, "feature_matrix_binned.csv"))
        print(f"  Bin-level features: {merged_bins.shape[1] - 2} bins")
        print(f"  Saved to {feat_dir}/feature_matrix_binned.csv")
    else:
        print("\nNo datasets processed successfully.")

    print("\nPipeline complete.")
