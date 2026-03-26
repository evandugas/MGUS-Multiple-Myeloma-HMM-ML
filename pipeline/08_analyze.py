#!/usr/bin/env python3
"""Post-hoc Analysis Suite
  - CNA landscape heatmap
  - Cross-dataset validation (pairwise and LODO)
  - ComBat batch correction (bin-level and arm-level)
  - Platform QC: bin coverage and noise by dataset
  - SHAP interpretability analysis
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))
import importlib
_gb = importlib.import_module("02_genomic_bins")
CENTROMERES = _gb.CENTROMERES
CHROMOSOMES = _gb.CHROMOSOMES
N_BINS = _gb.N_BINS

FEAT_DIR = os.path.join(BASE_DIR, "output", "features")
PLOT_DIR = os.path.join(BASE_DIR, "output", "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "output", "results")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42

ARM_ORDER = []
for i in range(1, 23):
    ARM_ORDER.append(f"chr{i}p")
    ARM_ORDER.append(f"chr{i}q")


# ============================================================================
# CNA Landscape Heatmap
# ============================================================================

def plot_cna_landscape(merged_df):
    """CNA landscape heatmap: rows=samples, cols=chromosomal arms."""
    print("\n--- CNA Landscape Heatmap ---")

    del_cols = sorted([c for c in merged_df.columns
                       if c.startswith("del_") and not c.startswith("seg_mean")])
    amp_cols = sorted([c for c in merged_df.columns
                       if c.startswith("amp_") and not c.startswith("seg_mean")])

    arm_del = {c.replace("del_", ""): c for c in del_cols}
    arm_amp = {c.replace("amp_", ""): c for c in amp_cols}
    arms = [a for a in ARM_ORDER if a in arm_del and a in arm_amp]

    net_cna = pd.DataFrame(index=merged_df.index)
    for arm in arms:
        net_cna[arm] = merged_df[arm_amp[arm]].fillna(0) - \
                        merged_df[arm_del[arm]].fillna(0)

    burden = net_cna.abs().mean(axis=1)

    mgus_idx = merged_df[merged_df["label"] == "MGUS"].index
    mm_idx = merged_df[merged_df["label"] == "MM"].index

    mgus_sorted = burden.loc[mgus_idx].sort_values().index
    mm_sorted = burden.loc[mm_idx].sort_values().index

    row_order = list(mgus_sorted) + list(mm_sorted)
    matrix = net_cna.loc[row_order, arms].values

    fig, ax = plt.subplots(figsize=(16, max(10, len(row_order) * 0.02 + 4)))
    vmax = 0.8
    ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
              interpolation="nearest")

    sep = len(mgus_sorted) - 0.5
    ax.axhline(sep, color="black", linewidth=2)

    ax.set_yticks([len(mgus_sorted) // 2, len(mgus_sorted) + len(mm_sorted) // 2])
    ax.set_yticklabels([f"MGUS (n={len(mgus_sorted)})",
                        f"MM (n={len(mm_sorted)})"], fontsize=12)

    ax.set_xticks(range(0, len(arms), 2))
    ax.set_xticklabels([arms[i].replace("chr", "").rstrip("pq")
                        for i in range(0, len(arms), 2)], fontsize=9)
    ax.set_xlabel("Chromosome", fontsize=12)

    for j in range(2, len(arms), 2):
        ax.axvline(j - 0.5, color="gray", linewidth=0.3, alpha=0.5)

    n_ds = merged_df["dataset"].nunique()
    ax.set_title(f"CNA Landscape: MGUS vs MM ({len(row_order)} samples, "
                 f"{n_ds} datasets)", fontsize=13)

    cbar = plt.colorbar(ax.images[0], ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Net CNA (amp - del fraction)", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "06_cna_landscape.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved 06_cna_landscape.png ({len(mgus_sorted)} MGUS + "
          f"{len(mm_sorted)} MM)")


# ============================================================================
# Platform QC
# ============================================================================

def plot_platform_qc(bin_df):
    """Platform QC: bin coverage and noise by dataset."""
    print("\n--- Platform QC ---")

    datasets = bin_df["dataset"].unique()
    bin_cols = [c for c in bin_df.columns if c not in ["label", "dataset"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Coverage: fraction of non-zero bins per dataset
    coverages = []
    noise_levels = []
    for ds in datasets:
        ds_data = bin_df[bin_df["dataset"] == ds][bin_cols].values
        # Non-zero = has data
        coverage = (ds_data != 0).mean(axis=1)
        coverages.append((ds, coverage.mean(), coverage.std()))
        # Noise: median absolute deviation of bin values per sample
        mads = np.median(np.abs(ds_data - np.median(ds_data, axis=1, keepdims=True)),
                         axis=1)
        noise_levels.append((ds, mads.mean(), mads.std()))

    ds_names = [c[0] for c in coverages]
    cov_means = [c[1] for c in coverages]
    cov_stds = [c[2] for c in coverages]
    noise_means = [n[1] for n in noise_levels]
    noise_stds = [n[2] for n in noise_levels]

    x = np.arange(len(ds_names))
    axes[0].bar(x, cov_means, yerr=cov_stds, capsize=4, alpha=0.85,
                color="steelblue")
    axes[0].set_ylabel("Fraction of bins with data")
    axes[0].set_title("Bin Coverage by Dataset")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ds_names, fontsize=8, rotation=30, ha="right")
    axes[0].set_ylim(0, 1.05)

    axes[1].bar(x, noise_means, yerr=noise_stds, capsize=4, alpha=0.85,
                color="coral")
    axes[1].set_ylabel("Median Absolute Deviation")
    axes[1].set_title("Per-Bin Noise by Dataset")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(ds_names, fontsize=8, rotation=30, ha="right")

    plt.suptitle("Platform QC: Bin-Level Data Quality", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "12_platform_qc.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved 12_platform_qc.png")


# ============================================================================
# Cross-Dataset Validation
# ============================================================================

def cross_dataset_validation(merged_df, feature_cols, label=""):
    """Pairwise cross-dataset validation: train on one, test on another."""
    print(f"\n--- Cross-Dataset Validation {label} ---")

    y = (merged_df["label"] == "MM").astype(int).values
    X = merged_df[feature_cols].fillna(0).values
    datasets = merged_df["dataset"].values

    unique_ds = sorted(merged_df["dataset"].unique())
    if len(unique_ds) < 2:
        print("  Only one dataset, skipping.")
        return None

    models = {
        "RF": RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "LR": LogisticRegression(
            C=1.0, solver="saga", max_iter=10000, l1_ratio=1.0,
            class_weight="balanced", random_state=RANDOM_STATE),
    }

    rows = []
    for train_ds in unique_ds:
        for test_ds in unique_ds:
            if train_ds == test_ds:
                continue

            train_mask = datasets == train_ds
            test_mask = datasets == test_ds

            X_tr, y_tr = X[train_mask], y[train_mask]
            X_te, y_te = X[test_mask], y[test_mask]

            if len(np.unique(y_te)) < 2 or len(np.unique(y_tr)) < 2:
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            for name, model in models.items():
                model.fit(X_tr_s, y_tr)
                y_prob = model.predict_proba(X_te_s)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                auc = roc_auc_score(y_te, y_prob)
                f1 = f1_score(y_te, y_pred, zero_division=0)
                bal_acc = balanced_accuracy_score(y_te, y_pred)

                row = {
                    "train_dataset": train_ds, "test_dataset": test_ds,
                    "model": name, "correction": label,
                    "AUC": auc, "F1": f1, "BalAcc": bal_acc,
                    "n_train": len(y_tr), "n_test": len(y_te),
                }
                rows.append(row)
                print(f"  {train_ds} -> {test_ds} ({name}): AUC={auc:.3f}  "
                      f"F1={f1:.3f}  BalAcc={bal_acc:.3f}")

    result_df = pd.DataFrame(rows)
    return result_df


# ============================================================================
# ComBat Batch Correction
# ============================================================================

def combat_batch_correction(df, feature_cols, batch_col="dataset"):
    """Apply ComBat batch correction. Returns corrected DataFrame."""
    print(f"\n--- ComBat Batch Correction ({len(feature_cols)} features) ---")

    try:
        from combat.pycombat import pycombat
    except ImportError:
        print("  combat not installed. Run: pip install combat")
        return None

    data = df[feature_cols].T
    batch = df[batch_col].values

    print(f"  Input: {data.shape[1]} samples x {data.shape[0]} features")
    print(f"  Batches: {dict(pd.Series(batch).value_counts())}")

    corrected = pycombat(data, batch)

    corrected_df = df.copy()
    corrected_df[feature_cols] = corrected.T.values
    print(f"  ComBat correction complete.")
    return corrected_df


# ============================================================================
# SHAP Analysis
# ============================================================================

def shap_analysis(merged_df, enhanced_cols):
    """SHAP analysis on HMM Enhanced RF model."""
    print("\n--- SHAP Analysis ---")

    try:
        import shap
    except ImportError:
        print("  SHAP not installed. Run: pip install shap")
        return

    X = merged_df[enhanced_cols].fillna(0).values
    y = (merged_df["label"] == "MM").astype(int).values
    feature_names = enhanced_cols

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=5, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)

    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Beeswarm
    print("  Generating beeswarm plot...")
    plt.figure(figsize=(12, 10))
    shap.summary_plot(sv, X, feature_names=feature_names, max_display=20,
                      show=False)
    plt.title("SHAP Feature Importance (HMM Enhanced RF)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "08_shap_beeswarm.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # Bar plot
    print("  Generating SHAP bar plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, X, feature_names=feature_names, plot_type="bar",
                      max_display=20, show=False)
    plt.title("Mean |SHAP Value| (HMM Enhanced RF)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "09_shap_bar.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # Waterfall for interesting cases
    print("  Generating waterfall plots...")
    y_prob = rf.predict_proba(X)[:, 1]

    mm_mask = y == 1
    mm_indices = np.where(mm_mask)[0]
    best_mm_idx = mm_indices[np.argmax(y_prob[mm_mask])]

    mgus_mask = y == 0
    mgus_indices = np.where(mgus_mask)[0]
    hardest_mgus_idx = mgus_indices[np.argmax(y_prob[mgus_mask])]

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]

    sample_id_mm = merged_df.index[best_mm_idx]
    sample_id_mgus = merged_df.index[hardest_mgus_idx]

    exp_mm = shap.Explanation(
        values=sv[best_mm_idx], base_values=base_val,
        data=X[best_mm_idx], feature_names=feature_names)
    exp_mgus = shap.Explanation(
        values=sv[hardest_mgus_idx], base_values=base_val,
        data=X[hardest_mgus_idx], feature_names=feature_names)

    fig = plt.figure(figsize=(12, 12))

    ax1 = fig.add_subplot(2, 1, 1)
    plt.sca(ax1)
    shap.plots.waterfall(exp_mm, max_display=15, show=False)
    ax1.set_title(f"Correct MM ({sample_id_mm}, P(MM)={y_prob[best_mm_idx]:.3f})",
                  fontsize=11)

    ax2 = fig.add_subplot(2, 1, 2)
    plt.sca(ax2)
    shap.plots.waterfall(exp_mgus, max_display=15, show=False)
    pred_label = "MM" if y_prob[hardest_mgus_idx] >= 0.5 else "MGUS"
    ax2.set_title(f"Hardest MGUS ({sample_id_mgus}, "
                  f"P(MM)={y_prob[hardest_mgus_idx]:.3f}, pred {pred_label})",
                  fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "10_shap_waterfall.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP plots")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Post-hoc Analysis Suite")
    print("=" * 60)

    # Load arm-level features
    merged = pd.read_csv(os.path.join(FEAT_DIR, "feature_matrix_arm_all.csv"),
                          index_col=0)
    merged = merged[merged["label"].isin(["MGUS", "MM"])]

    meta_cols = ["label", "dataset"]
    enhanced_cols = [c for c in merged.columns if c not in meta_cols]

    n_mgus = (merged["label"] == "MGUS").sum()
    n_mm = (merged["label"] == "MM").sum()
    n_ds = merged["dataset"].nunique()
    print(f"Loaded: {n_mgus} MGUS + {n_mm} MM = {len(merged)} samples "
          f"from {n_ds} datasets")
    print(f"Features: {len(enhanced_cols)}")

    # Load bin-level features
    bin_path = os.path.join(FEAT_DIR, "feature_matrix_binned.csv")
    bin_df = None
    if os.path.exists(bin_path):
        bin_df = pd.read_csv(bin_path, index_col=0)
        bin_df = bin_df[bin_df["label"].isin(["MGUS", "MM"])]
        bin_feat_cols = [c for c in bin_df.columns if c not in meta_cols]
        print(f"Bin-level features: {len(bin_feat_cols)}")

    # 1. CNA landscape
    plot_cna_landscape(merged)

    # 2. Platform QC (bin-level)
    if bin_df is not None:
        plot_platform_qc(bin_df)

    # 3. Cross-dataset validation (uncorrected, arm-level)
    cross_results = []
    result = cross_dataset_validation(merged, enhanced_cols, label="uncorrected")
    if result is not None:
        cross_results.append(result)

    # 4. ComBat on bin-level features (more features = better correction)
    if bin_df is not None:
        corrected_bins = combat_batch_correction(bin_df, bin_feat_cols)
        if corrected_bins is not None:
            corrected_bins.to_csv(os.path.join(FEAT_DIR,
                                  "feature_matrix_binned_combat.csv"))
            print(f"  Saved feature_matrix_binned_combat.csv")

    # 5. ComBat on arm-level features (for comparison)
    corrected_arm = combat_batch_correction(merged, enhanced_cols)
    if corrected_arm is not None:
        corrected_arm.to_csv(os.path.join(FEAT_DIR, "feature_matrix_combat.csv"))

        result = cross_dataset_validation(corrected_arm, enhanced_cols,
                                           label="ComBat-arm")
        if result is not None:
            cross_results.append(result)

    # 6. Standard CV on ComBat-corrected arm features
    if corrected_arm is not None:
        print("\n--- Standard CV on ComBat-corrected features ---")
        X_combat = corrected_arm[enhanced_cols].fillna(0).values
        y = (corrected_arm["label"] == "MM").astype(int).values

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10,
                                      random_state=RANDOM_STATE)
        models = {
            "RF": RandomForestClassifier(n_estimators=500, max_depth=5,
                    min_samples_leaf=3, class_weight="balanced",
                    random_state=RANDOM_STATE, n_jobs=-1),
            "LR": LogisticRegression(C=1.0, solver="saga", max_iter=10000,
                    l1_ratio=1.0, class_weight="balanced",
                    random_state=RANDOM_STATE),
        }

        for mname, model in models.items():
            aucs, f1s, bas = [], [], []
            for train_idx, test_idx in cv.split(X_combat, y):
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_combat[train_idx])
                X_te = scaler.transform(X_combat[test_idx])
                model.fit(X_tr, y[train_idx])
                y_prob = model.predict_proba(X_te)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)
                aucs.append(roc_auc_score(y[test_idx], y_prob))
                f1s.append(f1_score(y[test_idx], y_pred))
                bas.append(balanced_accuracy_score(y[test_idx], y_pred))
            print(f"  ComBat {mname}: AUC={np.mean(aucs):.3f}+/-{np.std(aucs):.3f}  "
                  f"F1={np.mean(f1s):.3f}  BalAcc={np.mean(bas):.3f}")

    # Save all cross-dataset results
    if cross_results:
        all_cross = pd.concat(cross_results, ignore_index=True)
        all_cross.to_csv(os.path.join(RESULTS_DIR, "cross_dataset_results.csv"),
                          index=False)
        print(f"\n  Saved cross_dataset_results.csv")

    # 7. SHAP analysis
    shap_analysis(merged, enhanced_cols)

    print("\n" + "=" * 60)
    print("  Analysis complete.")
    print("=" * 60)
