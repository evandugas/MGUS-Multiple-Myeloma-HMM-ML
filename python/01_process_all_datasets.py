#!/usr/bin/env python3
###############################################################################
# Unified data processing pipeline for all datasets
# Processes each dataset independently through:
#   1. Parse raw files (Agilent FE)
#   2. Clean (filter controls, drop sex chr, remove high-error probes, median-center)
#   3. Run HMM segmentation (with pooled transition matrix)
#   4. Extract arm-level CNA features + derived features
#   5. Export per-dataset results
# Then merges all arm-level features into one matrix.
#
# v2: Uses optimized HMM (pooled transmat, lower thresholds) and
#     extracts richer features (CNA burden, segment counts, segment means).
###############################################################################

import os
import sys
import gc
import gzip
import warnings
import csv
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

warnings.filterwarnings("ignore")

BASE_DIR = os.path.join("C:", os.sep, "Users", "Evan", "MGUS-Multiple-Myeloma-HMM-ML")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]
CENTROMERES = {
    "chr1": 125_000_000, "chr2": 93_300_000, "chr3": 91_000_000,
    "chr4": 50_400_000, "chr5": 48_400_000, "chr6": 61_000_000,
    "chr7": 59_900_000, "chr8": 45_600_000, "chr9": 49_000_000,
    "chr10": 40_200_000, "chr11": 53_700_000, "chr12": 35_800_000,
    "chr13": 17_900_000, "chr14": 17_600_000, "chr15": 19_000_000,
    "chr16": 36_600_000, "chr17": 22_200_000, "chr18": 18_200_000,
    "chr19": 26_500_000, "chr20": 27_500_000, "chr21": 13_200_000,
    "chr22": 14_700_000,
}
CNA_THRESHOLD = 0.20
N_WORKERS = 6

# Import HMM functions from existing module
sys.path.insert(0, os.path.join(BASE_DIR, "python"))
from hmm_core import (fit_hmm_chromosome, postprocess_states,
                       estimate_pooled_transmat, DEFAULT_MIN_CNA_PROBES)


# ============================================================================
# PARSERS
# ============================================================================

def parse_agilent_file(filepath, sample_id, label):
    """Parse an Agilent Feature Extraction .txt.gz file."""
    rows = []
    try:
        with gzip.open(filepath, "rt", errors="replace") as f:
            in_features = False
            header = None
            for line in f:
                if line.startswith("FEATURES"):
                    parts = line.strip().split("\t")
                    header = {col: i for i, col in enumerate(parts)}
                    in_features = True
                    continue
                if not in_features or header is None:
                    continue

                parts = line.strip().split("\t")
                if len(parts) < len(header):
                    continue

                try:
                    control_type = int(parts[header["ControlType"]])
                except (ValueError, KeyError):
                    continue

                if control_type != 0:
                    continue

                try:
                    log_ratio = float(parts[header["LogRatio"]])
                    log_ratio_err = float(parts[header["LogRatioError"]])
                    systematic_name = parts[header["SystematicName"]]
                except (ValueError, KeyError):
                    continue

                # Parse chr:start-end
                if ":" not in systematic_name or "-" not in systematic_name:
                    continue
                chrom, pos_range = systematic_name.split(":", 1)
                pos_parts = pos_range.split("-")
                if len(pos_parts) != 2:
                    continue
                try:
                    start = int(pos_parts[0])
                    end = int(pos_parts[1])
                except ValueError:
                    continue

                # Skip sex chromosomes and non-standard
                if chrom not in CHROMOSOMES:
                    continue

                rows.append((sample_id, label, chrom, start, end, log_ratio, log_ratio_err))
    except Exception as e:
        print(f"    ERROR parsing {filepath}: {e}")
        return None

    return rows


# ============================================================================
# PROCESSING PIPELINE
# ============================================================================

def clean_probe_data(rows_list):
    """Clean parsed probe data: remove high-error probes, median-center."""
    df = pd.DataFrame(rows_list,
                      columns=["sample_id", "label", "chr", "start", "end",
                               "LogRatio", "LogRatioError"])

    # Remove top 1% error probes
    err_thresh = df["LogRatioError"].quantile(0.99)
    df = df[df["LogRatioError"] <= err_thresh]

    # Median-center per sample
    medians = df.groupby("sample_id")["LogRatio"].transform("median")
    df["LogRatio"] = df["LogRatio"] - medians

    return df


def run_hmm_on_dataframe(probe_df):
    """Run HMM on a cleaned probe DataFrame with pooled transition matrices.

    Two-pass approach per sample:
      Pass 1: Fit each chromosome independently → estimate pooled transition matrix
      Pass 2: Re-fit each chromosome using the pooled transition matrix
    """
    executor = ProcessPoolExecutor(max_workers=N_WORKERS)
    all_states = []

    for sample_id, sample_df in probe_df.groupby("sample_id"):
        # Collect per-chromosome data
        chrom_data = {}
        for chrom in CHROMOSOMES:
            cdf = sample_df[sample_df["chr"] == chrom].sort_values("start")
            if len(cdf) < 20:
                continue
            lr = cdf["LogRatio"].values
            st = cdf["start"].values
            en = cdf["end"].values
            chrom_data[chrom] = (lr, st, en)

        if not chrom_data:
            continue

        # Pass 1: Estimate pooled transition matrix from this sample
        all_lr = {chrom: data[0] for chrom, data in chrom_data.items()}
        pooled_transmat = estimate_pooled_transmat(all_lr)

        # Pass 2: Re-fit with pooled transition matrix
        futures = {}
        for chrom, (lr, st, en) in chrom_data.items():
            futures[chrom] = executor.submit(
                fit_hmm_chromosome, (chrom, lr, st, en), pooled_transmat)

        for chrom, fut in futures.items():
            result = fut.result()
            if result is None:
                continue
            _, states, _, _ = result
            states = postprocess_states(states)
            lr, st, en = chrom_data[chrom]
            for i, state in enumerate(states):
                all_states.append((sample_id, chrom, int(st[i]),
                                   float(lr[i]), int(state)))

    executor.shutdown()
    return pd.DataFrame(all_states,
                        columns=["sample_id", "chr", "start", "log_ratio", "state"])


def compute_arm_features(probe_states_df, labels_dict):
    """Compute arm-level CNA features from probe states.

    Returns both basic arm fractions and derived features:
      - del_*/amp_* fractions per arm (88 features)
      - cna_burden: fraction of all probes in any CNA state
      - n_segments: total number of CNA segments
      - n_del_segments / n_amp_segments: segment counts by type
      - seg_count_*: number of CNA segments per arm
      - seg_mean_del_* / seg_mean_amp_*: mean log-ratio in CNA segments per arm
    """
    df = probe_states_df.copy()

    # Assign arms
    df["arm"] = "q"
    for chrom, centro in CENTROMERES.items():
        mask = (df["chr"] == chrom) & (df["start"] < centro)
        df.loc[mask, "arm"] = "p"
    df["arm_id"] = df["chr"] + df["arm"]

    # --- Basic arm fractions (same as before) ---
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

    # Pivot arm fractions
    del_frac = feat.pivot_table(index="sample_id", columns="arm_id",
                                 values="frac_del", fill_value=0)
    amp_frac = feat.pivot_table(index="sample_id", columns="arm_id",
                                 values="frac_amp", fill_value=0)
    del_frac.columns = [f"del_{c}" for c in del_frac.columns]
    amp_frac.columns = [f"amp_{c}" for c in amp_frac.columns]

    result = pd.concat([del_frac, amp_frac], axis=1)

    # --- Derived feature 1: CNA burden (fraction of genome in CNA state) ---
    sample_totals = df.groupby("sample_id").size()
    sample_cna = df[df["state"] != 1].groupby("sample_id").size()
    cna_burden = (sample_cna / sample_totals).fillna(0)
    result["cna_burden"] = result.index.map(cna_burden).fillna(0)

    # --- Derived feature 2: Segment counts ---
    # Count CNA segments per sample (contiguous runs of same non-neutral state)
    seg_counts = defaultdict(lambda: {"n_segments": 0, "n_del_segments": 0,
                                       "n_amp_segments": 0})
    arm_seg_counts = defaultdict(lambda: defaultdict(int))  # sample -> arm -> count
    arm_seg_means_del = defaultdict(lambda: defaultdict(list))  # sample -> arm -> [means]
    arm_seg_means_amp = defaultdict(lambda: defaultdict(list))

    for (sample_id, chrom), group in df.groupby(["sample_id", "chr"]):
        group = group.sort_values("start")
        states = group["state"].values
        log_ratios = group["log_ratio"].values
        starts = group["start"].values
        arms = group["arm_id"].values

        i = 0
        while i < len(states):
            if states[i] != 1:  # CNA state
                j = i
                while j < len(states) and states[j] == states[i]:
                    j += 1
                seg_counts[sample_id]["n_segments"] += 1
                seg_lr = log_ratios[i:j]
                seg_arm = arms[i]  # arm of segment start

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

    # Add global segment counts
    all_samples = result.index
    result["n_segments"] = [seg_counts[s]["n_segments"] for s in all_samples]
    result["n_del_segments"] = [seg_counts[s]["n_del_segments"] for s in all_samples]
    result["n_amp_segments"] = [seg_counts[s]["n_amp_segments"] for s in all_samples]

    # --- Derived feature 3: Per-arm segment counts ---
    all_arms = sorted(set(df["arm_id"]))
    for arm in all_arms:
        result[f"seg_count_{arm}"] = [arm_seg_counts[s].get(arm, 0) for s in all_samples]

    # --- Derived feature 4: Mean log-ratio within CNA segments per arm ---
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

    # Add label
    result.insert(0, "label", result.index.map(labels_dict))

    return result


def process_agilent_dataset(dataset_name, data_dir):
    """Process a complete Agilent dataset: parse → clean → HMM → arm features."""
    print(f"\n{'='*60}")
    print(f"  Processing {dataset_name} (Agilent)")
    print(f"{'='*60}")

    raw_dir = os.path.join(data_dir, "raw")
    labels_csv = os.path.join(data_dir, "sample_labels.csv")

    # Read labels
    labels_dict = {}
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            labels_dict[row["sample_id"]] = row["label"]

    # Remap mm_excluded to MM (we're merging all platforms at arm-level)
    labels_dict = {k: ("MM" if v == "mm_excluded" else v) for k, v in labels_dict.items()}

    # Collect files from mgus/ and mm/
    all_rows = []
    for subdir, label in [("mgus", "MGUS"), ("mm", "MM")]:
        dir_path = os.path.join(raw_dir, subdir)
        if not os.path.exists(dir_path):
            continue
        files = [f for f in os.listdir(dir_path) if f.endswith(".gz")]
        print(f"  {subdir}/: {len(files)} files")
        for filename in files:
            gsm_id = filename.split("_")[0].split(".")[0]
            filepath = os.path.join(dir_path, filename)
            rows = parse_agilent_file(filepath, gsm_id, label)
            if rows:
                all_rows.extend(rows)

    if not all_rows:
        print(f"  WARNING: No data parsed for {dataset_name}")
        return None

    print(f"  Parsed {len(all_rows):,} probe rows")

    # Clean
    probe_df = clean_probe_data(all_rows)
    del all_rows
    gc.collect()
    print(f"  After cleaning: {len(probe_df):,} rows, {probe_df['sample_id'].nunique()} samples")

    # HMM (with pooled transition matrix)
    print(f"  Running HMM (pooled transition matrix)...")
    probe_states = run_hmm_on_dataframe(probe_df)
    print(f"  HMM complete: {len(probe_states):,} probe states")

    # Arm features (basic + derived)
    arm_features = compute_arm_features(probe_states, labels_dict)
    n_basic = sum(1 for c in arm_features.columns if c.startswith(("del_", "amp_"))
                  and not c.startswith(("seg_mean_del", "seg_mean_amp")))
    n_derived = arm_features.shape[1] - n_basic - 1  # -1 for label
    print(f"  Features: {n_basic} arm fractions + {n_derived} derived = {arm_features.shape[1] - 1} total")

    # Save per-dataset
    out_dir = os.path.join(data_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)
    arm_features.to_csv(os.path.join(out_dir, "arm_features.csv"))
    print(f"  Saved to {out_dir}/arm_features.csv")

    return arm_features


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Agilent Dataset Processing Pipeline (v2 — optimized HMM)")
    print("=" * 60)

    all_arm_features = []

    for ds in ["GSE77975", "GSE33685"]:
        ds_dir = os.path.join(DATA_DIR, ds)
        if os.path.exists(ds_dir):
            result = process_agilent_dataset(ds, ds_dir)
            if result is not None:
                all_arm_features.append((ds, result))

    # --- Merge all arm-level features ---
    if all_arm_features:
        print(f"\n{'='*60}")
        print(f"  Merging {len(all_arm_features)} datasets")
        print(f"{'='*60}")

        merged_parts = []
        for ds_name, df in all_arm_features:
            df = df.copy()
            df.insert(1, "dataset", ds_name)
            merged_parts.append(df)
            mgus = (df["label"] == "MGUS").sum()
            mm = (df["label"] == "MM").sum()
            print(f"  {ds_name}: {mgus} MGUS + {mm} MM = {len(df)}")

        merged = pd.concat(merged_parts, ignore_index=False)

        # Ensure all arm columns exist (fill missing with 0)
        merged = merged.fillna(0)

        feat_dir = os.path.join(OUTPUT_DIR, "features")
        os.makedirs(feat_dir, exist_ok=True)
        merged.to_csv(os.path.join(feat_dir, "feature_matrix_arm_all.csv"))

        total_mgus = (merged["label"] == "MGUS").sum()
        total_mm = (merged["label"] == "MM").sum()
        n_feat = merged.shape[1] - 2  # excluding label and dataset
        print(f"\n  MERGED: {total_mgus} MGUS + {total_mm} MM = {len(merged)} samples")
        print(f"  Features: {n_feat} total")
        print(f"  Saved to {feat_dir}/feature_matrix_arm_all.csv")
    else:
        print("\nNo datasets processed successfully.")

    print("\nPipeline complete.")
