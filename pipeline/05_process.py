#!/usr/bin/env python3
"""Unified data processing pipeline for GSE77975 + GSE33685.

Processes each dataset independently through:
  1. Parse raw Agilent files — PARALLEL
  2. Clean (filter controls, drop sex chr, remove high-error probes, median-center)
  3. Bin probes into fixed 1Mb genomic bins — PARALLEL per sample
  4. Run HMM segmentation (with pooled transition matrix) — PARALLEL per sample
  5. Extract arm-level CNA features + derived features
  6. Export per-dataset results
Then merges all arm-level features into one master matrix.
"""

import os
import sys
import gc
import warnings
import csv
import logging
import shutil
import subprocess
import tempfile
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

# Suppress HMM convergence warnings globally
warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

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
adaptive_emission_params = _hmm_core.adaptive_emission_params
DEFAULT_MIN_CNA_PROBES = _hmm_core.DEFAULT_MIN_CNA_PROBES

CHROMOSOMES = _genomic_bins.CHROMOSOMES
CENTROMERES = _genomic_bins.CENTROMERES
N_BINS = _genomic_bins.N_BINS
BIN_TABLE = _genomic_bins.BIN_TABLE
compute_bin_values = _genomic_bins.compute_bin_values
binned_to_chrom_arrays = _genomic_bins.binned_to_chrom_arrays
bins_to_arm_mapping = _genomic_bins.bins_to_arm_mapping

parse_agilent_file = _parsers.parse_agilent_file

N_WORKERS = min(8, os.cpu_count() or 4)

# Find Rscript for CBS
RSCRIPT = shutil.which("Rscript")
if not RSCRIPT:
    for rver in sorted(os.listdir("C:/Program Files/R"), reverse=True) \
            if os.path.isdir("C:/Program Files/R") else []:
        candidate = os.path.join("C:/Program Files/R", rver, "bin", "Rscript.exe")
        if os.path.isfile(candidate):
            RSCRIPT = candidate
            break
CBS_SCRIPT = os.path.join(BASE_DIR, "pipeline", "10_cbs_segment.R")

# Dataset registry (Agilent only)
DATASET_CONFIG = {
    "GSE77975": {"platform": "agilent"},
    "GSE33685": {"platform": "agilent"},
}


# ============================================================================
# WORKER FUNCTIONS (top-level for pickling)
# ============================================================================

def _parse_one_file(args):
    """Worker: parse a single Agilent raw file. Returns (sample_id, rows) or (sample_id, None)."""
    filepath, gsm_id, label = args
    warnings.filterwarnings("ignore")
    rows = parse_agilent_file(filepath, gsm_id, label)
    return gsm_id, rows


def _bin_one_sample(args):
    """Worker: bin probe data for one sample. Returns (sample_id, bin_values)."""
    sample_id, chroms, starts, log_ratios = args
    warnings.filterwarnings("ignore")

    # Build a mini DataFrame for compute_bin_values
    df = pd.DataFrame({"chr": chroms, "start": starts, "LogRatio": log_ratios})
    bin_values = compute_bin_values(df)

    return sample_id, bin_values


def _hmm_one_sample(args):
    """Worker: run two-pass HMM on one sample's binned data.

    Returns dict with:
      'states': list of (sample_id, chrom, midpoint, log_ratio, state, bin_idx)
      'smoothed': dict of bin_idx -> smoothed log2 ratio (emission mean of assigned state)
      'posteriors': dict of bin_idx -> (P_del, P_amp)
    """
    sample_id, bin_values = args
    warnings.filterwarnings("ignore")
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)

    chrom_data = binned_to_chrom_arrays(bin_values)
    if not chrom_data:
        return {"states": [], "smoothed": {}, "posteriors": {}}

    # Adaptive emission initialization from this sample's data
    all_lr = {chrom: data[0] for chrom, data in chrom_data.items()
              if len(data[0]) >= 10}
    if not all_lr:
        return {"states": [], "smoothed": {}, "posteriors": {}}
    init_means, init_covars = adaptive_emission_params(all_lr)

    # Pass 1: Estimate pooled transition matrix (with adaptive init)
    pooled_transmat = estimate_pooled_transmat(all_lr, init_means, init_covars)

    # Pass 2: Fit each chromosome with pooled transmat + adaptive init
    state_results = []
    smoothed = {}
    posteriors = {}
    for chrom, (lr, midpoints, indices) in chrom_data.items():
        if len(lr) < 10:
            continue
        ends = midpoints + 500_000
        result = fit_hmm_chromosome((chrom, lr, midpoints, ends), pooled_transmat,
                                    init_means, init_covars)
        if result is None:
            continue
        _, states, sorted_means, _, post_probs = result
        states = postprocess_states(states)
        for i, state in enumerate(states):
            bin_idx = int(indices[i])
            state_results.append((sample_id, chrom, int(midpoints[i]),
                                  float(lr[i]), int(state), bin_idx))
            # Smoothed: replace raw value with emission mean of assigned state
            smoothed[bin_idx] = float(sorted_means[state])
            # Posteriors: P(del), P(amp) from forward-backward
            posteriors[bin_idx] = (float(post_probs[i, 0]),
                                   float(post_probs[i, 2]))

    return {"states": state_results, "smoothed": smoothed, "posteriors": posteriors}


# ============================================================================
# PROCESSING PIPELINE
# ============================================================================

def clean_probe_data(rows_list):
    """Clean parsed probe data: remove high-error probes, median-center."""
    df = pd.DataFrame(rows_list,
                      columns=["sample_id", "label", "chr", "start", "end",
                               "LogRatio", "LogRatioError"])

    # Remove high-error probes
    if df["LogRatioError"].max() > 0:
        err_thresh = df["LogRatioError"].quantile(0.99)
        df = df[df["LogRatioError"] <= err_thresh]

    # Median-center per sample
    medians = df.groupby("sample_id")["LogRatio"].transform("median")
    df["LogRatio"] = df["LogRatio"] - medians

    return df


def run_cbs_segmentation(probe_df):
    """Run CBS segmentation on probe-level data via R/DNAcopy.

    Args:
        probe_df: DataFrame with columns [sample_id, chr, start, LogRatio]

    Returns:
        DataFrame with CBS segments: sample_id, chr, start, end, n_probes, seg_mean
        or None if R/DNAcopy unavailable.
    """
    if not RSCRIPT or not os.path.isfile(CBS_SCRIPT):
        print("  CBS unavailable (Rscript or 10_cbs_segment.R not found)", flush=True)
        return None

    # Write probe data to temp file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".tsv", delete=False,
                                      dir=os.path.join(BASE_DIR, "data")) as f:
        tmp_input = f.name
        probe_df[["sample_id", "chr", "start", "LogRatio"]].to_csv(
            f, sep="\t", index=False)

    tmp_output = tmp_input.replace(".tsv", "_segments.tsv")

    try:
        result = subprocess.run(
            [RSCRIPT, CBS_SCRIPT, tmp_input, tmp_output],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            print(f"  CBS failed (exit {result.returncode}):", flush=True)
            print(result.stderr[-500:] if result.stderr else "(no stderr)", flush=True)
            return None
        print(result.stdout.strip(), flush=True)

        segments = pd.read_csv(tmp_output, sep="\t")
        return segments
    except subprocess.TimeoutExpired:
        print("  CBS timed out (>60min)", flush=True)
        return None
    finally:
        for f in [tmp_input, tmp_output]:
            if os.path.exists(f):
                os.unlink(f)


def compute_cbs_arm_features(segments_df, labels_dict):
    """Extract arm-level features from CBS segments.

    For each arm, computes:
      - Weighted mean segment value (weighted by segment length)
      - Weighted SD
      - Number of segments (complexity)
      - Max absolute segment mean (most extreme event)

    Returns DataFrame with 4 features per arm.
    """
    df = segments_df.copy()

    # Assign arms
    df["arm"] = "q"
    for chrom, centro in CENTROMERES.items():
        mask = (df["chr"] == chrom) & (df["start"] < centro)
        df.loc[mask, "arm"] = "p"
    df["arm_id"] = df["chr"] + df["arm"]
    df["seg_length"] = df["end"] - df["start"]

    all_arms = sorted(set(df["arm_id"]))
    sample_ids = sorted(df["sample_id"].unique())

    rows = []
    for sid in sample_ids:
        sdf = df[df["sample_id"] == sid]
        row = {"sample_id": sid, "label": labels_dict.get(sid, "")}

        for arm_id in all_arms:
            arm_segs = sdf[sdf["arm_id"] == arm_id]
            if len(arm_segs) > 0:
                lengths = arm_segs["seg_length"].values.astype(float)
                means = arm_segs["seg_mean"].values.astype(float)
                total_len = lengths.sum()

                if total_len > 0:
                    # Weighted mean
                    wmean = np.average(means, weights=lengths)
                    # Weighted SD
                    wvar = np.average((means - wmean) ** 2, weights=lengths)
                    wsd = np.sqrt(wvar)
                else:
                    wmean = 0.0
                    wsd = 0.0

                row[f"cbs_mean_{arm_id}"] = wmean
                row[f"cbs_sd_{arm_id}"] = wsd
                row[f"cbs_nseg_{arm_id}"] = len(arm_segs)
                row[f"cbs_maxabs_{arm_id}"] = float(np.max(np.abs(means)))
            else:
                row[f"cbs_mean_{arm_id}"] = 0.0
                row[f"cbs_sd_{arm_id}"] = 0.0
                row[f"cbs_nseg_{arm_id}"] = 0
                row[f"cbs_maxabs_{arm_id}"] = 0.0

        rows.append(row)

    result = pd.DataFrame(rows).set_index("sample_id")
    return result


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
    print(f"\n{'='*60}")
    print(f"  Processing {dataset_name} (Agilent)")
    print(f"{'='*60}", flush=True)

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
    labels_dict = {k: ("MM" if v == "mm_excluded" else v)
                   for k, v in labels_dict.items()}

    # Collect file list from mgus/ and mm/
    file_list = []
    for subdir, label in [("mgus", "MGUS"), ("mm", "MM")]:
        dir_path = os.path.join(raw_dir, subdir)
        if not os.path.exists(dir_path):
            continue
        files = sorted([f for f in os.listdir(dir_path) if f.endswith(".gz")])
        print(f"  {subdir}/: {len(files)} files", flush=True)
        for filename in files:
            gsm_id = filename.split("_")[0].split(".")[0]
            filepath = os.path.join(dir_path, filename)
            file_list.append((filepath, gsm_id, label))

    if not file_list:
        print(f"  WARNING: No files found for {dataset_name}")
        return None

    # --- STEP 1: Parallel file parsing ---
    print(f"  Parsing {len(file_list)} files ({N_WORKERS} workers)...", flush=True)
    all_rows = []
    n_parsed = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_parse_one_file, args): args[1]
                   for args in file_list}
        for future in as_completed(futures):
            gsm_id, rows = future.result()
            if rows:
                all_rows.extend(rows)
                n_parsed += 1

    print(f"  Parsed {n_parsed} samples, {len(all_rows):,} probe rows", flush=True)

    if not all_rows:
        print(f"  WARNING: No data parsed for {dataset_name}")
        return None

    # --- STEP 2: Clean ---
    probe_df = clean_probe_data(all_rows)
    del all_rows
    gc.collect()
    n_samples = probe_df["sample_id"].nunique()
    print(f"  After cleaning: {len(probe_df):,} rows, {n_samples} samples", flush=True)

    # --- STEP 2b: CBS segmentation on probe-level data ---
    print(f"  Running CBS segmentation (probe-level, via R/DNAcopy)...", flush=True)
    cbs_segments = run_cbs_segmentation(probe_df)
    cbs_features = None
    if cbs_segments is not None:
        n_segs = len(cbs_segments)
        n_cbs_samples = cbs_segments["sample_id"].nunique()
        print(f"  CBS complete: {n_segs:,} segments across {n_cbs_samples} samples",
              flush=True)
        cbs_features = compute_cbs_arm_features(cbs_segments, labels_dict)
        print(f"  CBS features: {cbs_features.shape[1] - 1} "
              f"(4 per arm)", flush=True)

    # --- STEP 3: Parallel binning ---
    print(f"  Binning into 1Mb bins ({N_BINS} bins/sample)...", flush=True)
    bin_args = []
    for sample_id, sdf in probe_df.groupby("sample_id"):
        bin_args.append((sample_id, sdf["chr"].values, sdf["start"].values,
                         sdf["LogRatio"].values))
    del probe_df
    gc.collect()

    sample_bin_values = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_bin_one_sample, args): args[0]
                   for args in bin_args}
        for future in as_completed(futures):
            sample_id, bin_vals = future.result()
            sample_bin_values[sample_id] = bin_vals
    del bin_args
    gc.collect()

    n_binned = len(sample_bin_values)
    mean_bins = np.mean([np.sum(~np.isnan(v)) for v in sample_bin_values.values()])
    print(f"  Binned: {n_binned} samples, avg {mean_bins:.0f} bins/sample", flush=True)

    # --- STEP 4: Parallel HMM ---
    print(f"  Running HMM ({N_WORKERS} workers)...", flush=True)
    hmm_args = [(sid, bv) for sid, bv in sample_bin_values.items()]

    all_states = []
    all_smoothed = {}  # sample_id -> {bin_idx: smoothed_value}
    all_posteriors = {}  # sample_id -> {bin_idx: (p_del, p_amp)}
    n_done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(_hmm_one_sample, args): args[0]
                   for args in hmm_args}
        for future in as_completed(futures):
            hmm_result = future.result()
            all_states.extend(hmm_result["states"])
            sid = hmm_result["states"][0][0] if hmm_result["states"] else futures[future]
            all_smoothed[sid] = hmm_result["smoothed"]
            all_posteriors[sid] = hmm_result["posteriors"]
            n_done += 1
            if n_done % 50 == 0:
                print(f"    HMM: {n_done}/{n_binned} samples done", flush=True)

    print(f"  HMM complete: {len(all_states):,} bin states", flush=True)

    probe_states = pd.DataFrame(all_states,
                                columns=["sample_id", "chr", "start", "log_ratio",
                                         "state", "bin_idx"])

    # --- STEP 5: Arm features ---
    arm_features = compute_arm_features(probe_states, labels_dict)
    n_basic = sum(1 for c in arm_features.columns if c.startswith(("del_", "amp_"))
                  and not c.startswith(("seg_mean_del", "seg_mean_amp")))
    n_derived = arm_features.shape[1] - n_basic - 1
    print(f"  Features: {n_basic} arm fractions + {n_derived} derived = "
          f"{arm_features.shape[1] - 1} total", flush=True)

    # --- STEP 5b: Smoothed and posterior arm features ---
    arm_map = bins_to_arm_mapping()
    all_arms = sorted(set(arm_map.values()))

    smoothed_rows = []
    posterior_rows = []
    for sid in arm_features.index:
        s_smooth = all_smoothed.get(sid, {})
        s_post = all_posteriors.get(sid, {})
        label = labels_dict.get(sid, "")

        smooth_row = {"sample_id": sid, "label": label}
        post_row = {"sample_id": sid, "label": label}

        for arm_id in all_arms:
            arm_bins = [idx for idx, arm in arm_map.items() if arm == arm_id]

            # Smoothed: mean + SD of HMM-smoothed log2 ratios per arm
            s_vals = [s_smooth[b] for b in arm_bins if b in s_smooth]
            if s_vals:
                smooth_row[f"mean_{arm_id}"] = float(np.mean(s_vals))
                smooth_row[f"sd_{arm_id}"] = float(np.std(s_vals)) if len(s_vals) > 1 else 0.0
            else:
                smooth_row[f"mean_{arm_id}"] = 0.0
                smooth_row[f"sd_{arm_id}"] = 0.0

            # Posterior: mean P(del) and mean P(amp) per arm
            p_vals = [s_post[b] for b in arm_bins if b in s_post]
            if p_vals:
                post_row[f"p_del_{arm_id}"] = float(np.mean([p[0] for p in p_vals]))
                post_row[f"p_amp_{arm_id}"] = float(np.mean([p[1] for p in p_vals]))
            else:
                post_row[f"p_del_{arm_id}"] = 0.0
                post_row[f"p_amp_{arm_id}"] = 0.0

        smoothed_rows.append(smooth_row)
        posterior_rows.append(post_row)

    smoothed_df = pd.DataFrame(smoothed_rows).set_index("sample_id")
    posterior_df = pd.DataFrame(posterior_rows).set_index("sample_id")
    print(f"  Smoothed features: {smoothed_df.shape[1] - 1} "
          f"({len(all_arms)} arms x 2)", flush=True)
    print(f"  Posterior features: {posterior_df.shape[1] - 1} "
          f"({len(all_arms)} arms x 2)", flush=True)

    # Save per-dataset
    out_dir = os.path.join(data_dir, "processed")
    os.makedirs(out_dir, exist_ok=True)
    arm_features.to_csv(os.path.join(out_dir, "arm_features.csv"))
    smoothed_df.to_csv(os.path.join(out_dir, "smoothed_arm_features.csv"))
    posterior_df.to_csv(os.path.join(out_dir, "posterior_arm_features.csv"))

    # Also save bin-level data
    bin_rows = []
    for sid, bv in sample_bin_values.items():
        for i in range(N_BINS):
            if not np.isnan(bv[i]):
                bin_rows.append((sid, i, float(bv[i])))
    bin_df = pd.DataFrame(bin_rows, columns=["sample_id", "bin_idx", "log_ratio"])
    bin_matrix = bin_df.pivot_table(index="sample_id", columns="bin_idx",
                                    values="log_ratio", fill_value=0)
    bin_matrix.to_csv(os.path.join(out_dir, "bin_features.csv"))
    if cbs_features is not None:
        cbs_features.to_csv(os.path.join(out_dir, "cbs_arm_features.csv"))
    print(f"  Saved to {out_dir}/", flush=True)

    return arm_features, bin_matrix, smoothed_df, posterior_df, cbs_features


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Processing Pipeline (GSE77975 + GSE33685)")
    print("=" * 60, flush=True)

    all_arm_features = []
    all_bin_features = []
    all_smoothed_features = []
    all_posterior_features = []
    all_cbs_features = []

    for ds_name in DATASET_CONFIG:
        ds_dir = os.path.join(DATA_DIR, ds_name)
        if not os.path.exists(ds_dir):
            print(f"\n  Skipping {ds_name} (directory not found)")
            continue
        result = process_dataset(ds_name, ds_dir)
        if result is not None:
            arm_feat, bin_feat, smooth_feat, post_feat, cbs_feat = result
            all_arm_features.append((ds_name, arm_feat))
            all_bin_features.append((ds_name, bin_feat))
            all_smoothed_features.append((ds_name, smooth_feat))
            if cbs_feat is not None:
                all_cbs_features.append((ds_name, cbs_feat))
            all_posterior_features.append((ds_name, post_feat))

    # --- Merge all features ---
    if all_arm_features:
        print(f"\n{'='*60}")
        print(f"  Merging {len(all_arm_features)} datasets")
        print(f"{'='*60}", flush=True)

        feat_dir = os.path.join(OUTPUT_DIR, "features")
        os.makedirs(feat_dir, exist_ok=True)

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
            arm_labels = all_arm_features[[n for n, _ in all_arm_features].index(ds_name)][1]["label"]
            df.insert(0, "label", df.index.map(arm_labels))
            bin_parts.append(df)

        merged_bins = pd.concat(bin_parts, ignore_index=False)
        merged_bins = merged_bins.fillna(0)
        merged_bins.to_csv(os.path.join(feat_dir, "feature_matrix_binned.csv"))
        print(f"  Bin-level features: {merged_bins.shape[1] - 2} bins")
        print(f"  Saved to {feat_dir}/feature_matrix_binned.csv")

        # Merge smoothed arm features
        smooth_parts = []
        for ds_name, df in all_smoothed_features:
            df = df.copy()
            df.insert(1, "dataset", ds_name)
            smooth_parts.append(df)
        merged_smooth = pd.concat(smooth_parts, ignore_index=False).fillna(0)
        merged_smooth.to_csv(os.path.join(feat_dir, "feature_matrix_smoothed_arm.csv"))
        print(f"  Smoothed arm features: {merged_smooth.shape[1] - 2}")
        print(f"  Saved to {feat_dir}/feature_matrix_smoothed_arm.csv")

        # Merge posterior arm features
        post_parts = []
        for ds_name, df in all_posterior_features:
            df = df.copy()
            df.insert(1, "dataset", ds_name)
            post_parts.append(df)
        merged_post = pd.concat(post_parts, ignore_index=False).fillna(0)
        merged_post.to_csv(os.path.join(feat_dir, "feature_matrix_posterior_arm.csv"))
        print(f"  Posterior arm features: {merged_post.shape[1] - 2}")
        print(f"  Saved to {feat_dir}/feature_matrix_posterior_arm.csv")

        # Merge CBS arm features
        if all_cbs_features:
            cbs_parts = []
            for ds_name, df in all_cbs_features:
                df = df.copy()
                df.insert(1, "dataset", ds_name)
                cbs_parts.append(df)
            merged_cbs = pd.concat(cbs_parts, ignore_index=False).fillna(0)
            merged_cbs.to_csv(os.path.join(feat_dir, "feature_matrix_cbs_arm.csv"))
            print(f"  CBS arm features: {merged_cbs.shape[1] - 2}")
            print(f"  Saved to {feat_dir}/feature_matrix_cbs_arm.csv")
    else:
        print("\nNo datasets processed successfully.")

    print("\nPipeline complete.", flush=True)
