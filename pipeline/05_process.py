#!/usr/bin/env python3
"""Data processing pipeline for GSE77975.

  1. Parse raw Agilent files
  2. Clean probes (filter errors, median-center)
  3. Stage 1: Train cohort-wide 3-state HMM
  4. Stage 2: Per-sample Viterbi decoding
  5. Extract arm-level features (HMM + Raw)
"""

import os, sys, gc, warnings, csv, logging
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "GSE77975")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))
import importlib
_parsers = importlib.import_module("03_parsers")
_hmm = importlib.import_module("04_hmm_core")

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

N_WORKERS = min(8, os.cpu_count() or 4)


def _parse_one_file(args):
    filepath, gsm_id, label = args
    warnings.filterwarnings("ignore")
    return gsm_id, _parsers.parse_agilent_file(filepath, gsm_id, label)


def _decode_one_sample(args):
    sample_id, chrom_data, n_states, means, covars, transmat = args
    warnings.filterwarnings("ignore")
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    return _hmm.decode_sample(chrom_data, n_states, means, covars, transmat, sample_id)


def clean_probes(rows_list):
    df = pd.DataFrame(rows_list,
                      columns=["sample_id", "label", "chr", "start", "end",
                               "LogRatio", "LogRatioError"])
    if df["LogRatioError"].max() > 0:
        df = df[df["LogRatioError"] <= df["LogRatioError"].quantile(0.99)]
    medians = df.groupby("sample_id")["LogRatio"].transform("median")
    df["LogRatio"] = df["LogRatio"] - medians
    return df


def assign_arm(chrom, start):
    return "p" if start < CENTROMERES.get(chrom, 0) else "q"


def compute_hmm_features(probe_states_df, labels_dict):
    """Arm-level CNA features from Viterbi states."""
    df = probe_states_df.copy()
    df["arm_id"] = df.apply(lambda r: r["chr"] + assign_arm(r["chr"], r["start"]), axis=1)

    # Del/amp fractions per arm
    total = df.groupby(["sample_id", "arm_id"]).size().reset_index(name="n")
    n_del = df[df["state"] == 0].groupby(["sample_id", "arm_id"]).size().reset_index(name="n_del")
    n_amp = df[df["state"] == 2].groupby(["sample_id", "arm_id"]).size().reset_index(name="n_amp")

    feat = total.merge(n_del, on=["sample_id", "arm_id"], how="left")
    feat = feat.merge(n_amp, on=["sample_id", "arm_id"], how="left")
    feat["n_del"] = feat["n_del"].fillna(0)
    feat["n_amp"] = feat["n_amp"].fillna(0)
    feat["frac_del"] = feat["n_del"] / feat["n"]
    feat["frac_amp"] = feat["n_amp"] / feat["n"]

    del_piv = feat.pivot_table(index="sample_id", columns="arm_id", values="frac_del", fill_value=0)
    amp_piv = feat.pivot_table(index="sample_id", columns="arm_id", values="frac_amp", fill_value=0)
    del_piv.columns = [f"del_{c}" for c in del_piv.columns]
    amp_piv.columns = [f"amp_{c}" for c in amp_piv.columns]
    result = pd.concat([del_piv, amp_piv], axis=1)

    # CNA burden
    sample_n = df.groupby("sample_id").size()
    sample_cna = df[df["state"] != 1].groupby("sample_id").size()
    result["cna_burden"] = result.index.map((sample_cna / sample_n).fillna(0)).fillna(0)

    # Segment counts + means
    seg = defaultdict(lambda: {"n_seg": 0, "n_del_seg": 0, "n_amp_seg": 0})
    arm_seg_count = defaultdict(lambda: defaultdict(int))
    arm_seg_mean_del = defaultdict(lambda: defaultdict(list))
    arm_seg_mean_amp = defaultdict(lambda: defaultdict(list))

    for (sid, chrom), grp in df.groupby(["sample_id", "chr"]):
        grp = grp.sort_values("start")
        states = grp["state"].values
        lr = grp["log_ratio"].values
        arms = grp["arm_id"].values
        i = 0
        while i < len(states):
            if states[i] != 1:
                j = i
                while j < len(states) and states[j] == states[i]:
                    j += 1
                seg[sid]["n_seg"] += 1
                a = arms[i]
                if states[i] == 0:
                    seg[sid]["n_del_seg"] += 1
                    arm_seg_count[sid][a] += 1
                    arm_seg_mean_del[sid][a].append(float(np.mean(lr[i:j])))
                else:
                    seg[sid]["n_amp_seg"] += 1
                    arm_seg_count[sid][a] += 1
                    arm_seg_mean_amp[sid][a].append(float(np.mean(lr[i:j])))
                i = j
            else:
                i += 1

    samples = result.index

    # Normalize segment counts by probe count per sample
    # (removes platform density confound: 60K arrays produce more segments than 1M)
    sample_probe_counts = df.groupby("sample_id").size()
    for s in samples:
        n_probes = sample_probe_counts.get(s, 1)
        # Segments per 10K probes — comparable across platforms
        scale = 10000.0 / n_probes
        seg[s]["n_seg"] *= scale
        seg[s]["n_del_seg"] *= scale
        seg[s]["n_amp_seg"] *= scale
        for arm in arm_seg_count[s]:
            arm_seg_count[s][arm] *= scale

    result["n_segments"] = [seg[s]["n_seg"] for s in samples]
    result["n_del_segments"] = [seg[s]["n_del_seg"] for s in samples]
    result["n_amp_segments"] = [seg[s]["n_amp_seg"] for s in samples]

    all_arms = sorted(set(df["arm_id"]))
    for arm in all_arms:
        result[f"seg_count_{arm}"] = [arm_seg_count[s].get(arm, 0) for s in samples]
    for arm in all_arms:
        result[f"seg_mean_del_{arm}"] = [
            float(np.mean(arm_seg_mean_del[s][arm])) if arm_seg_mean_del[s].get(arm) else 0.0
            for s in samples]
        result[f"seg_mean_amp_{arm}"] = [
            float(np.mean(arm_seg_mean_amp[s][arm])) if arm_seg_mean_amp[s].get(arm) else 0.0
            for s in samples]

    result.insert(0, "label", result.index.map(labels_dict))
    return result


def compute_raw_features(probe_df, labels_dict):
    """Mean + SD of raw log2R per arm from probe data."""
    df = probe_df[["sample_id", "chr", "start", "LogRatio"]].copy()
    df["arm_id"] = df.apply(lambda r: r["chr"] + assign_arm(r["chr"], r["start"]), axis=1)

    grp = df.groupby(["sample_id", "arm_id"])["LogRatio"]
    mean_piv = grp.mean().reset_index(name="v").pivot_table(
        index="sample_id", columns="arm_id", values="v", fill_value=0)
    sd_piv = grp.std().fillna(0).reset_index(name="v").pivot_table(
        index="sample_id", columns="arm_id", values="v", fill_value=0)
    mean_piv.columns = [f"mean_{c}" for c in mean_piv.columns]
    sd_piv.columns = [f"sd_{c}" for c in sd_piv.columns]

    result = pd.concat([mean_piv, sd_piv], axis=1)
    result.insert(0, "label", result.index.map(labels_dict))
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  Pipeline: GSE77975 — Cohort-trained HMM (BIC selection)")
    print("=" * 60, flush=True)

    labels_dict = {}
    with open(os.path.join(DATA_DIR, "sample_labels.csv")) as f:
        for row in csv.DictReader(f):
            labels_dict[row["sample_id"]] = row["label"]

    # Parse
    file_list = []
    for subdir, label in [("mgus", "MGUS"), ("mm", "MM")]:
        d = os.path.join(DATA_DIR, "raw", subdir)
        if not os.path.exists(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".gz"))
        print(f"  {subdir}/: {len(files)} files", flush=True)
        for fn in files:
            gsm = fn.split("_")[0].split(".")[0]
            file_list.append((os.path.join(d, fn), gsm, label))

    print(f"\n  Parsing {len(file_list)} files...", flush=True)
    all_rows = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        for future in as_completed({ex.submit(_parse_one_file, a): a[1] for a in file_list}):
            gsm, rows = future.result()
            if rows:
                all_rows.extend(rows)
    print(f"  {len(all_rows):,} probe rows", flush=True)

    # Clean
    probe_df = clean_probes(all_rows)
    del all_rows; gc.collect()
    print(f"  After cleaning: {len(probe_df):,} rows, "
          f"{probe_df['sample_id'].nunique()} samples", flush=True)

    # Organize per-sample per-chromosome
    sample_chrom = {}
    all_seqs = []
    for sid, sdf in probe_df.groupby("sample_id"):
        cd = {}
        for chrom in CHROMOSOMES:
            mask = sdf["chr"] == chrom
            if mask.sum() < 20:
                continue
            cdf = sdf[mask].sort_values("start")
            lr = cdf["LogRatio"].values
            starts = cdf["start"].values
            cd[chrom] = (lr, starts)
            all_seqs.append(lr)
        sample_chrom[sid] = cd

    # Stage 0: BIC model selection
    print(f"\n  Stage 0: BIC model selection...", flush=True)
    n_states = _hmm.select_n_states(all_seqs)

    # Stage 1: Cohort training
    print(f"\n  Stage 1: Cohort HMM training...", flush=True)
    n_states, means, covars, transmat = _hmm.train_cohort_model(all_seqs, n_states)
    del all_seqs; gc.collect()

    # Stage 2: Per-sample decoding
    print(f"\n  Stage 2: Decoding {len(sample_chrom)} samples...", flush=True)
    args = [(sid, cd, n_states, means, covars, transmat) for sid, cd in sample_chrom.items()]
    all_states = []
    n_done = 0
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = {ex.submit(_decode_one_sample, a): a[0] for a in args}
        for f in as_completed(futs):
            all_states.extend(f.result())
            n_done += 1
            if n_done % 20 == 0:
                print(f"    {n_done}/{len(sample_chrom)} done", flush=True)
    del args, sample_chrom; gc.collect()
    print(f"  {len(all_states):,} probe states", flush=True)

    states_df = pd.DataFrame(all_states,
                             columns=["sample_id", "chr", "start", "log_ratio", "state"])
    del all_states; gc.collect()

    # Features
    print(f"\n  Computing features...", flush=True)
    hmm_feat = compute_hmm_features(states_df, labels_dict)
    del states_df; gc.collect()
    raw_feat = compute_raw_features(probe_df, labels_dict)
    del probe_df; gc.collect()

    # Save
    feat_dir = os.path.join(OUTPUT_DIR, "features")
    os.makedirs(feat_dir, exist_ok=True)
    hmm_feat.to_csv(os.path.join(feat_dir, "feature_matrix_hmm.csv"))
    raw_feat.to_csv(os.path.join(feat_dir, "feature_matrix_raw.csv"))

    n_mgus = (hmm_feat["label"] == "MGUS").sum()
    n_mm = (hmm_feat["label"] == "MM").sum()
    print(f"\n  {n_mgus} MGUS + {n_mm} MM = {len(hmm_feat)} samples")
    print(f"  HMM features: {hmm_feat.shape[1] - 1}")
    print(f"  Raw features: {raw_feat.shape[1] - 1}")
    print("  Done.", flush=True)
