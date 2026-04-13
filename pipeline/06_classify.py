#!/usr/bin/env python3
"""Phase 4: ML Classification

Compares four feature sets for MGUS/MM classification:
  1. HMM arm fractions only (baseline)
  2. HMM enhanced (arm fractions + derived: burden, segments, means)
  3. Raw bin-level arm stats (mean + SD per arm from 1Mb bins)
  4. HMM + Raw combined
Includes: confusion matrices, per-class metrics, statistical significance,
          leave-one-dataset-out cross-validation.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score,
                             roc_curve, confusion_matrix, recall_score)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))
import importlib
_gb = importlib.import_module("02_genomic_bins")
CENTROMERES = _gb.CENTROMERES
CHROMOSOMES = _gb.CHROMOSOMES
BIN_TABLE = _gb.BIN_TABLE
bins_to_arm_mapping = _gb.bins_to_arm_mapping

FEAT_DIR = os.path.join(BASE_DIR, "output", "features")
PLOT_DIR = os.path.join(BASE_DIR, "output", "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "output", "results")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42


def make_models():
    """Create fresh model instances."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Logistic Regression L1": LogisticRegression(
            C=1.0, solver="saga", max_iter=10000, l1_ratio=1.0,
            class_weight="balanced", random_state=RANDOM_STATE),
    }


def run_cv(X, y, feature_names, dataset_name):
    """Run repeated stratified k-fold CV with RF and LR."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                  random_state=RANDOM_STATE)

    models = make_models()
    results = {name: {"auc": [], "f1": [], "bal_acc": [],
                       "y_true_all": [], "y_prob_all": [], "y_pred_all": [],
                       "importances": []}
               for name in models}

    for train_idx, test_idx in cv.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        for name, model in models.items():
            if "Logistic" in name:
                model.fit(X_tr_s, y_tr)
                y_prob = model.predict_proba(X_te_s)[:, 1]
            else:
                model.fit(X_tr, y_tr)
                y_prob = model.predict_proba(X_te)[:, 1]

            y_pred = (y_prob >= 0.5).astype(int)
            results[name]["auc"].append(roc_auc_score(y_te, y_prob))
            results[name]["f1"].append(f1_score(y_te, y_pred))
            results[name]["bal_acc"].append(balanced_accuracy_score(y_te, y_pred))
            results[name]["y_true_all"].extend(y_te.tolist())
            results[name]["y_prob_all"].extend(y_prob.tolist())
            results[name]["y_pred_all"].extend(y_pred.tolist())

            if "Random Forest" in name:
                results[name]["importances"].append(model.feature_importances_)
            elif "Logistic" in name:
                results[name]["importances"].append(np.abs(model.coef_[0]))

    print(f"\n  === {dataset_name} ===")
    summary_rows = []
    for name in models:
        r = results[name]
        y_true_arr = np.array(r["y_true_all"])
        y_pred_arr = np.array(r["y_pred_all"])
        sensitivity = recall_score(y_true_arr, y_pred_arr, pos_label=1)
        specificity = recall_score(y_true_arr, y_pred_arr, pos_label=0)

        row = {
            "model": name, "dataset": dataset_name,
            "AUC_mean": np.mean(r["auc"]), "AUC_std": np.std(r["auc"]),
            "F1_mean": np.mean(r["f1"]), "F1_std": np.std(r["f1"]),
            "BalAcc_mean": np.mean(r["bal_acc"]), "BalAcc_std": np.std(r["bal_acc"]),
            "Sensitivity": sensitivity, "Specificity": specificity,
        }
        summary_rows.append(row)
        print(f"  {name}: AUC={row['AUC_mean']:.3f}+/-{row['AUC_std']:.3f}  "
              f"F1={row['F1_mean']:.3f}  BalAcc={row['BalAcc_mean']:.3f}  "
              f"Sens={sensitivity:.3f}  Spec={specificity:.3f}")

    for name in models:
        imps = np.array(results[name]["importances"])
        results[name]["mean_importance"] = imps.mean(axis=0)

    return results, summary_rows


def run_lodo_cv(X, y, datasets, feature_names, feat_set_name):
    """Leave-one-dataset-out cross-validation.

    Train on all datasets except one, test on the held-out dataset.
    Returns per-dataset results and summary rows.
    """
    unique_ds = sorted(datasets.unique())
    if len(unique_ds) < 2:
        print("  LODO requires >= 2 datasets, skipping")
        return [], []

    models = make_models()
    all_rows = []

    print(f"\n  === Leave-One-Dataset-Out: {feat_set_name} ===")

    for test_ds in unique_ds:
        test_mask = datasets == test_ds
        train_mask = ~test_mask

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        if len(np.unique(y_te)) < 2:
            # Test set has only one class — can't compute AUC
            n_mgus = (y_te == 0).sum()
            n_mm = (y_te == 1).sum()
            print(f"  {test_ds}: {n_mgus} MGUS + {n_mm} MM — skipped (single class)")
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        for name, model in models.items():
            model_fresh = make_models()[name]
            if "Logistic" in name:
                model_fresh.fit(X_tr_s, y_tr)
                y_prob = model_fresh.predict_proba(X_te_s)[:, 1]
            else:
                model_fresh.fit(X_tr, y_tr)
                y_prob = model_fresh.predict_proba(X_te)[:, 1]

            y_pred = (y_prob >= 0.5).astype(int)
            auc = roc_auc_score(y_te, y_prob)
            f1 = f1_score(y_te, y_pred, zero_division=0)
            ba = balanced_accuracy_score(y_te, y_pred)

            row = {
                "test_dataset": test_ds,
                "model": name,
                "features": feat_set_name,
                "n_train": len(y_tr),
                "n_test": len(y_te),
                "AUC": auc, "F1": f1, "BalAcc": ba,
            }
            all_rows.append(row)
            print(f"  Test={test_ds} ({len(y_te)} samples), {name}: "
                  f"AUC={auc:.3f}  F1={f1:.3f}  BalAcc={ba:.3f}")

    return all_rows


def statistical_tests(all_results):
    """Paired t-test and Wilcoxon on per-fold metrics between feature sets.

    Structured as:
      Q1 (core):    HMM Basic vs Raw — does HMM discretization help?
      Q2 (derived): HMM Enhanced vs HMM Basic — do derived features add value?
      Q3 (combined): Combined vs Raw — does adding HMM help raw?
      Q4 (combined): Combined vs HMM Enhanced — does adding raw help HMM?
    """
    comparisons = [
        ("CBS", "Raw", "Q1: CBS denoising vs raw (core question)"),
        ("CBS Full", "Raw", "Q2: CBS full features vs raw"),
        ("CBS", "HMM Basic", "Q3: CBS vs HMM discretization"),
        ("CBS", "HMM Posterior", "Q4: CBS vs HMM posteriors"),
        ("HMM Basic", "Raw", "Q5: HMM discretization vs raw"),
        ("HMM Posterior", "Raw", "Q6: HMM posteriors vs raw"),
        ("HMM Posterior", "HMM Basic", "Q7: posteriors vs discrete HMM"),
    ]
    model = "Random Forest"
    rows = []

    print("\n  === Statistical Significance (RF, paired across CV folds) ===")
    for ds_a, ds_b, question in comparisons:
        if ds_a not in all_results or ds_b not in all_results:
            continue
        print(f"\n  {question}")
        for metric in ["auc", "f1", "bal_acc"]:
            a = np.array(all_results[ds_a][model][metric])
            b = np.array(all_results[ds_b][model][metric])
            diff = np.mean(a) - np.mean(b)
            t_stat, t_pval = stats.ttest_rel(a, b)
            w_stat, w_pval = stats.wilcoxon(a, b)
            rows.append({
                "comparison": f"{ds_a} vs {ds_b}",
                "question": question,
                "metric": metric.upper(),
                "mean_diff": diff,
                "t_statistic": t_stat, "t_pvalue": t_pval,
                "wilcoxon_statistic": w_stat, "wilcoxon_pvalue": w_pval,
            })
            sig = "***" if t_pval < 0.001 else "**" if t_pval < 0.01 \
                else "*" if t_pval < 0.05 else "ns"
            winner = ds_a if diff > 0 else ds_b
            print(f"    {metric.upper()}: diff={diff:+.4f}  p={t_pval:.4f} {sig}"
                  f"  ({winner} wins)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "statistical_tests.csv"), index=False)
    return df


def plot_confusion_matrices(all_results, dataset_names):
    """Plot confusion matrix heatmaps for RF across all feature sets."""
    model = "Random Forest"
    n = len(dataset_names)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, ds_name in zip(axes, dataset_names):
        r = all_results[ds_name][model]
        y_true = np.array(r["y_true_all"])
        y_pred = np.array(r["y_pred_all"])
        cm = confusion_matrix(y_true, y_pred)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["MGUS", "MM"], fontsize=10)
        ax.set_yticklabels(["MGUS", "MM"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)

        short = _short_name(ds_name)
        ax.set_title(f"{short} RF", fontsize=12)

        for i in range(2):
            for j in range(2):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                        ha="center", va="center", color=color, fontsize=10)

    plt.suptitle("Confusion Matrices (RF, pooled across CV folds)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "05_confusion_matrices.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


def build_raw_arm_features(merged_df, bin_matrix=None):
    """Build raw arm-level features (mean + SD per arm).

    If bin_matrix is provided, computes from bin-level log2 ratios
    (platform-agnostic). Otherwise falls back to re-parsing raw files.
    """
    arm_map = bins_to_arm_mapping()
    all_arms = sorted(set(arm_map.values()))

    if bin_matrix is not None:
        # Platform-agnostic: compute from bin-level data
        rows = []
        for sample_id in merged_df.index:
            if sample_id not in bin_matrix.index:
                continue
            row = {"sample_id": sample_id, "label": merged_df.loc[sample_id, "label"]}
            bin_vals = bin_matrix.loc[sample_id]

            for arm_id in all_arms:
                arm_bins = [str(idx) for idx, arm in arm_map.items() if arm == arm_id]
                arm_cols = [c for c in arm_bins if c in bin_vals.index]
                if arm_cols:
                    vals = bin_vals[arm_cols].values.astype(float)
                    vals = vals[~np.isnan(vals)]  # skip NaN bins only
                    if len(vals) > 0:
                        row[f"mean_{arm_id}"] = float(np.mean(vals))
                        row[f"sd_{arm_id}"] = float(np.std(vals)) if len(vals) > 1 else 0.0
                    else:
                        row[f"mean_{arm_id}"] = 0.0
                        row[f"sd_{arm_id}"] = 0.0
                else:
                    row[f"mean_{arm_id}"] = 0.0
                    row[f"sd_{arm_id}"] = 0.0
            rows.append(row)

        result = pd.DataFrame(rows).set_index("sample_id")
        return result

    # Fallback: re-parse Agilent raw files (only works for Agilent datasets)
    import gzip
    sample_ids = merged_df.index.tolist()
    labels = dict(zip(merged_df.index, merged_df["label"]))
    datasets_col = dict(zip(merged_df.index, merged_df["dataset"]))

    ds_samples = {}
    for sid in sample_ids:
        ds = datasets_col[sid]
        if ds not in ds_samples:
            ds_samples[ds] = []
        ds_samples[ds].append(sid)

    arm_stats = {sid: {} for sid in sample_ids}

    for ds_name, sids in ds_samples.items():
        sid_set = set(sids)
        raw_dir = os.path.join(BASE_DIR, "data", ds_name, "raw")

        for subdir in ["mgus", "mm"]:
            dir_path = os.path.join(raw_dir, subdir)
            if not os.path.exists(dir_path):
                continue

            for filename in os.listdir(dir_path):
                if not filename.endswith(".gz"):
                    continue
                gsm_id = filename.split("_")[0].split(".")[0]
                if gsm_id not in sid_set:
                    continue

                # Only parse Agilent files (have FEATURES header)
                filepath = os.path.join(dir_path, filename)
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
                            try:
                                ct = int(parts[header["ControlType"]])
                                if ct != 0:
                                    continue
                                lr = float(parts[header["LogRatio"]])
                                sn = parts[header["SystematicName"]]
                            except (ValueError, KeyError, IndexError):
                                continue

                            if ":" not in sn or "-" not in sn:
                                continue
                            chrom, pos_range = sn.split(":", 1)
                            if chrom not in CHROMOSOMES:
                                continue
                            try:
                                start = int(pos_range.split("-")[0])
                            except ValueError:
                                continue

                            arm = "p" if start < CENTROMERES.get(chrom, 0) else "q"
                            arm_id = chrom + arm

                            if arm_id not in arm_stats[gsm_id]:
                                arm_stats[gsm_id][arm_id] = [0.0, 0.0, 0]
                            arm_stats[gsm_id][arm_id][0] += lr
                            arm_stats[gsm_id][arm_id][1] += lr * lr
                            arm_stats[gsm_id][arm_id][2] += 1
                except Exception:
                    continue

        print(f"  Parsed raw LogRatios for {ds_name}")

    all_arms_found = sorted(set(a for sid in arm_stats for a in arm_stats[sid]))
    rows = []
    for sid in sample_ids:
        row = {"sample_id": sid, "label": labels[sid]}
        for arm_id in all_arms_found:
            s_val, sq, n = arm_stats[sid].get(arm_id, [0, 0, 0])
            mean = s_val / n if n > 0 else 0
            var = (sq / n - mean ** 2) if n > 1 else 0
            row[f"mean_{arm_id}"] = mean
            row[f"sd_{arm_id}"] = np.sqrt(max(var, 0))
        rows.append(row)

    result = pd.DataFrame(rows).set_index("sample_id")
    return result


def _short_name(ds_name):
    names = {
        "HMM Basic": "HMM",
        "HMM Enhanced": "HMM+",
        "Raw": "Raw",
        "Combined": "Comb",
        "HMM Smoothed": "Smooth",
        "HMM Posterior": "Post",
        "CBS": "CBS",
        "CBS Full": "CBS+",
    }
    return names.get(ds_name, ds_name)


if __name__ == "__main__":
    print("=" * 60)
    print("  Classification: HMM vs Raw Features")
    print("=" * 60)

    # Load merged HMM features
    merged = pd.read_csv(os.path.join(FEAT_DIR, "feature_matrix_arm_all.csv"),
                          index_col=0)
    merged = merged[merged["label"].isin(["MGUS", "MM"])]

    y = (merged["label"] == "MM").astype(int).values
    ds_labels = merged["dataset"]

    # Load bin-level features for raw arm computation
    bin_path = os.path.join(FEAT_DIR, "feature_matrix_binned.csv")
    bin_matrix = None
    if os.path.exists(bin_path):
        bin_matrix = pd.read_csv(bin_path, index_col=0)
        # Drop metadata columns
        meta = [c for c in ["label", "dataset"] if c in bin_matrix.columns]
        if meta:
            bin_matrix = bin_matrix.drop(columns=meta)

    # Separate feature sets
    meta_cols = ["label", "dataset"]
    all_feat_cols = [c for c in merged.columns if c not in meta_cols]

    basic_cols = [c for c in all_feat_cols
                  if (c.startswith("del_") or c.startswith("amp_"))
                  and not c.startswith(("seg_mean_del_", "seg_mean_amp_"))]
    enhanced_cols = all_feat_cols

    # Get the arm set from HMM basic features (41 arms)
    hmm_arms = sorted(set(c.replace("del_", "").replace("amp_", "")
                          for c in basic_cols))

    X_basic = merged[basic_cols].fillna(0).values
    X_enhanced = merged[enhanced_cols].fillna(0).values

    n_mgus = (y == 0).sum()
    n_mm = (y == 1).sum()
    n_datasets = ds_labels.nunique()
    print(f"Loaded: {n_mgus} MGUS + {n_mm} MM = {len(y)} samples "
          f"from {n_datasets} datasets")
    print(f"HMM basic features: {X_basic.shape[1]} ({len(hmm_arms)} arms x 2)")
    print(f"HMM enhanced features: {X_enhanced.shape[1]}")

    # Build raw arm features (platform-agnostic from bins)
    print("\nBuilding raw arm features from bin-level data...")
    raw_df = build_raw_arm_features(merged, bin_matrix=bin_matrix)
    raw_df.to_csv(os.path.join(FEAT_DIR, "feature_matrix_raw_arm.csv"))

    # Align raw features to same arm set as HMM for fair comparison
    raw_aligned_cols = []
    for arm in hmm_arms:
        for prefix in ["mean_", "sd_"]:
            col = f"{prefix}{arm}"
            if col in raw_df.columns:
                raw_aligned_cols.append(col)
    X_raw = raw_df[raw_aligned_cols].fillna(0).values
    n_raw_total = len([c for c in raw_df.columns if c != "label"])
    print(f"Raw features (aligned to HMM arms): {X_raw.shape[1]} ({len(hmm_arms)} arms x 2)")
    if n_raw_total != len(raw_aligned_cols):
        print(f"  (dropped {n_raw_total - len(raw_aligned_cols)} features "
              f"from non-HMM arms for fair comparison)")

    # Load HMM Smoothed features (mean + SD of HMM-denoised log2 ratios per arm)
    smooth_path = os.path.join(FEAT_DIR, "feature_matrix_smoothed_arm.csv")
    X_smoothed = None
    smoothed_cols = []
    if os.path.exists(smooth_path):
        smooth_df = pd.read_csv(smooth_path, index_col=0)
        smooth_df = smooth_df.loc[merged.index]
        # Align to HMM arm set
        smoothed_cols = []
        for arm in hmm_arms:
            for prefix in ["mean_", "sd_"]:
                col = f"{prefix}{arm}"
                if col in smooth_df.columns:
                    smoothed_cols.append(col)
        X_smoothed = smooth_df[smoothed_cols].fillna(0).values
        print(f"HMM Smoothed features: {X_smoothed.shape[1]} ({len(hmm_arms)} arms x 2)")

    # Load HMM Posterior features (mean P(del), P(amp) per arm)
    post_path = os.path.join(FEAT_DIR, "feature_matrix_posterior_arm.csv")
    X_posterior = None
    posterior_cols = []
    if os.path.exists(post_path):
        post_df = pd.read_csv(post_path, index_col=0)
        post_df = post_df.loc[merged.index]
        posterior_cols = []
        for arm in hmm_arms:
            for prefix in ["p_del_", "p_amp_"]:
                col = f"{prefix}{arm}"
                if col in post_df.columns:
                    posterior_cols.append(col)
        X_posterior = post_df[posterior_cols].fillna(0).values
        print(f"HMM Posterior features: {X_posterior.shape[1]} ({len(hmm_arms)} arms x 2)")

    # Load CBS features (probe-level CBS segmentation, continuous)
    cbs_path = os.path.join(FEAT_DIR, "feature_matrix_cbs_arm.csv")
    X_cbs = None
    cbs_cols = []
    cbs_mean_cols = []
    if os.path.exists(cbs_path):
        cbs_df = pd.read_csv(cbs_path, index_col=0)
        cbs_df = cbs_df.loc[merged.index]
        # CBS mean+SD (same format as Raw for fair comparison)
        cbs_mean_cols = []
        for arm in hmm_arms:
            for prefix in ["cbs_mean_", "cbs_sd_"]:
                col = f"{prefix}{arm}"
                if col in cbs_df.columns:
                    cbs_mean_cols.append(col)
        X_cbs = cbs_df[cbs_mean_cols].fillna(0).values
        # CBS full (mean + SD + nseg + maxabs)
        cbs_cols = [c for c in cbs_df.columns if c not in ["label", "dataset"]]
        X_cbs_full = cbs_df[cbs_cols].fillna(0).values
        print(f"CBS features (mean+SD): {X_cbs.shape[1]} ({len(hmm_arms)} arms x 2)")
        print(f"CBS features (full): {X_cbs_full.shape[1]} ({len(hmm_arms)} arms x 4)")

    # Combined: HMM Enhanced + Raw
    X_combined = np.hstack([X_enhanced, X_raw])
    combined_feat_names = enhanced_cols + raw_aligned_cols
    print(f"Combined features: {X_combined.shape[1]}")

    # =========================================================================
    # Standard repeated stratified CV
    # =========================================================================
    feature_sets = [
        ("HMM Basic", X_basic, basic_cols),
        ("HMM Enhanced", X_enhanced, enhanced_cols),
        ("Raw", X_raw, raw_aligned_cols),
        ("Combined", X_combined, combined_feat_names),
    ]
    if X_smoothed is not None:
        feature_sets.append(("HMM Smoothed", X_smoothed, smoothed_cols))
    if X_posterior is not None:
        feature_sets.append(("HMM Posterior", X_posterior, posterior_cols))
    if X_cbs is not None:
        feature_sets.append(("CBS", X_cbs, cbs_mean_cols))
        feature_sets.append(("CBS Full", X_cbs_full, cbs_cols))

    all_results = {}
    all_summaries = []

    for name, X, feat_names in feature_sets:
        print(f"\nRunning CV: {name} ({X.shape[1]} features)...")
        results, summaries = run_cv(X, y, feat_names, name)
        all_results[name] = results
        all_summaries.extend(summaries)

    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "classification_results.csv"),
                       index=False)

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(summary_df[["dataset", "model", "AUC_mean", "AUC_std",
                       "F1_mean", "BalAcc_mean", "Sensitivity",
                       "Specificity"]].to_string(index=False))

    # Statistical significance
    stat_df = statistical_tests(all_results)

    # =========================================================================
    # Leave-one-dataset-out CV
    # =========================================================================
    print("\n" + "=" * 60)
    print("  LEAVE-ONE-DATASET-OUT CV")
    print("=" * 60)

    lodo_rows = []
    for name, X, feat_names in feature_sets:
        rows = run_lodo_cv(X, y, ds_labels, feat_names, name)
        lodo_rows.extend(rows)

    if lodo_rows:
        lodo_df = pd.DataFrame(lodo_rows)
        lodo_df.to_csv(os.path.join(RESULTS_DIR, "lodo_results.csv"), index=False)
        print(f"\n  LODO results saved to {RESULTS_DIR}/lodo_results.csv")

    # =========================================================================
    # Plots
    # =========================================================================
    print("\nGenerating plots...")
    dataset_names = summary_df["dataset"].unique()
    model_names = summary_df["model"].unique()
    x_pos = np.arange(len(dataset_names))
    width = 0.35
    bar_colors = ["steelblue", "coral"]
    short_labels = [_short_name(d) for d in dataset_names]

    # Plot 1: AUC comparison
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, model in enumerate(model_names):
        subset = summary_df[summary_df["model"] == model]
        means = [subset[subset["dataset"] == d]["AUC_mean"].values[0]
                 for d in dataset_names]
        stds = [subset[subset["dataset"] == d]["AUC_std"].values[0]
                for d in dataset_names]
        ax.bar(x_pos + i * width, means, width, yerr=stds, label=model,
               capsize=4, alpha=0.85, color=bar_colors[i])
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title(f"Classification: All Feature Sets ({len(y)} samples, "
                 f"{n_datasets} datasets)", fontsize=13)
    ax.set_xticks(x_pos + width / 2)
    ax.set_xticklabels(short_labels, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "01_classification_auc.png"), dpi=150)
    plt.close()

    # Plot 2: ROC curves
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_configs = [
        ("HMM Basic", "Random Forest", "steelblue", "-"),
        ("HMM Basic", "Logistic Regression L1", "steelblue", "--"),
        ("HMM Enhanced", "Random Forest", "forestgreen", "-"),
        ("HMM Enhanced", "Logistic Regression L1", "forestgreen", "--"),
        ("Raw", "Random Forest", "firebrick", "-"),
        ("Raw", "Logistic Regression L1", "firebrick", "--"),
        ("Combined", "Random Forest", "darkorange", "-"),
        ("Combined", "Logistic Regression L1", "darkorange", "--"),
        ("HMM Smoothed", "Random Forest", "darkorchid", "-"),
        ("HMM Smoothed", "Logistic Regression L1", "darkorchid", "--"),
        ("HMM Posterior", "Random Forest", "teal", "-"),
        ("HMM Posterior", "Logistic Regression L1", "teal", "--"),
        ("CBS", "Random Forest", "deeppink", "-"),
        ("CBS", "Logistic Regression L1", "deeppink", "--"),
        ("CBS Full", "Random Forest", "gold", "-"),
        ("CBS Full", "Logistic Regression L1", "gold", "--"),
    ]
    # Only plot feature sets that were actually evaluated
    plot_configs = [(n, m, c, ls) for n, m, c, ls in plot_configs
                    if n in all_results]

    for ds_name, model_name, color, ls in plot_configs:
        r = all_results[ds_name][model_name]
        y_true = np.array(r["y_true_all"])
        y_prob = np.array(r["y_prob_all"])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        label_short = _short_name(ds_name)
        model_short = "RF" if "Random" in model_name else "LR"
        ax.plot(fpr, tpr, color=color, linewidth=2, linestyle=ls,
                label=f"{label_short} {model_short} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves: All Feature Sets", fontsize=13)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_roc_curves.png"), dpi=150)
    plt.close()

    # Plot 3: Feature importance (HMM Enhanced RF)
    fig, ax = plt.subplots(figsize=(10, 8))
    imp = all_results["HMM Enhanced"]["Random Forest"]["mean_importance"]
    top_idx = np.argsort(imp)[-20:]
    top_names = [enhanced_cols[i] for i in top_idx]
    top_vals = imp[top_idx]

    def feature_color(name):
        if name.startswith("del_") or name.startswith("seg_mean_del_"):
            return "steelblue"
        elif name.startswith("amp_") or name.startswith("seg_mean_amp_"):
            return "firebrick"
        return "goldenrod"

    colors_bar = [feature_color(n) for n in top_names]
    ax.barh(range(len(top_names)), top_vals, color=colors_bar, alpha=0.85)
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names, fontsize=9)
    ax.set_xlabel("Mean Feature Importance (RF)", fontsize=12)
    ax.set_title("Top 20 Predictive Features (HMM Enhanced)", fontsize=13)
    legend_elements = [
        Line2D([0], [0], color="steelblue", lw=8, label="Deletion"),
        Line2D([0], [0], color="firebrick", lw=8, label="Amplification"),
        Line2D([0], [0], color="goldenrod", lw=8, label="Derived"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_feature_importance.png"), dpi=150)
    plt.close()

    # Plot 4: All metrics comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = [("AUC_mean", "AUC_std", "AUC-ROC"),
               ("F1_mean", "F1_std", "F1 Score"),
               ("BalAcc_mean", "BalAcc_std", "Balanced Accuracy")]

    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        for i, model in enumerate(model_names):
            subset = summary_df[summary_df["model"] == model]
            means = [subset[subset["dataset"] == d][mean_col].values[0]
                     for d in dataset_names]
            stds = [subset[subset["dataset"] == d][std_col].values[0]
                    for d in dataset_names]
            ax.bar(x_pos + i * width, means, width, yerr=stds, label=model,
                   capsize=4, alpha=0.85, color=bar_colors[i])
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x_pos + width / 2)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
        if ax == axes[0]:
            ax.legend(fontsize=8)

    plt.suptitle(f"All Metrics: 4 Feature Sets ({len(y)} samples)", fontsize=13,
                 y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "04_all_metrics.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # Plot 5: Confusion matrices
    plot_confusion_matrices(all_results, dataset_names)

    # Plot 6: LODO results
    if lodo_rows:
        lodo_df = pd.DataFrame(lodo_rows)
        fig, ax = plt.subplots(figsize=(14, 6))
        feat_sets = lodo_df["features"].unique()
        test_datasets = lodo_df["test_dataset"].unique()
        x_lodo = np.arange(len(test_datasets))
        w = 0.8 / len(feat_sets)

        for i, fs in enumerate(feat_sets):
            fs_data = lodo_df[(lodo_df["features"] == fs) &
                              (lodo_df["model"] == "Random Forest")]
            aucs = []
            for td in test_datasets:
                row = fs_data[fs_data["test_dataset"] == td]
                aucs.append(row["AUC"].values[0] if len(row) > 0 else 0)
            ax.bar(x_lodo + i * w, aucs, w, label=_short_name(fs), alpha=0.85)

        ax.set_ylabel("AUC-ROC", fontsize=12)
        ax.set_title("Leave-One-Dataset-Out CV (RF)", fontsize=13)
        ax.set_xticks(x_lodo + w * len(feat_sets) / 2)
        ax.set_xticklabels(test_datasets, fontsize=9, rotation=30, ha="right")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, "11_lodo_cv.png"), dpi=150,
                    bbox_inches="tight")
        plt.close()

    print(f"\nPlots saved to {PLOT_DIR}")
    print(f"Results saved to {RESULTS_DIR}")
    print("Classification complete.")
