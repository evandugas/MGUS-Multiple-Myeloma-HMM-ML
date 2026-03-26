###############################################################################
# Post-hoc Analysis Suite
#   - CNA landscape heatmap
#   - Cross-dataset validation
#   - SHAP interpretability analysis
###############################################################################

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score,
                             recall_score)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.join("C:", os.sep, "Users", "Evan", "MGUS-Multiple-Myeloma-HMM-ML")
FEAT_DIR = os.path.join(BASE_DIR, "output", "features")
PLOT_DIR = os.path.join(BASE_DIR, "output", "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "output", "results")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42

# Canonical arm order for heatmap
ARM_ORDER = []
for i in range(1, 23):
    chrom = f"chr{i}"
    ARM_ORDER.append(f"{chrom}p")
    ARM_ORDER.append(f"{chrom}q")


# ============================================================================
# Enhancement 5: CNA Landscape Heatmap
# ============================================================================

def plot_cna_landscape(merged_df):
    """CNA landscape heatmap: rows=samples, cols=chromosomal arms."""
    print("\n--- CNA Landscape Heatmap ---")

    # Get del/amp fraction columns (not seg_mean)
    del_cols = sorted([c for c in merged_df.columns
                       if c.startswith("del_") and not c.startswith("seg_mean")])
    amp_cols = sorted([c for c in merged_df.columns
                       if c.startswith("amp_") and not c.startswith("seg_mean")])

    # Build arm -> del/amp column mapping
    arm_del = {c.replace("del_", ""): c for c in del_cols}
    arm_amp = {c.replace("amp_", ""): c for c in amp_cols}

    # Available arms in canonical order
    arms = [a for a in ARM_ORDER if a in arm_del and a in arm_amp]

    # Compute net CNA: amp - del (positive = amplification, negative = deletion)
    net_cna = pd.DataFrame(index=merged_df.index)
    for arm in arms:
        net_cna[arm] = merged_df[arm_amp[arm]].fillna(0) - merged_df[arm_del[arm]].fillna(0)

    # Compute CNA burden for sorting
    burden = net_cna.abs().mean(axis=1)

    # Split by label, sort by burden within each
    mgus_idx = merged_df[merged_df["label"] == "MGUS"].index
    mm_idx = merged_df[merged_df["label"] == "MM"].index

    mgus_sorted = burden.loc[mgus_idx].sort_values().index
    mm_sorted = burden.loc[mm_idx].sort_values().index

    row_order = list(mgus_sorted) + list(mm_sorted)
    matrix = net_cna.loc[row_order, arms].values

    # Plot
    fig, ax = plt.subplots(figsize=(16, 10))
    vmax = 0.8
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")

    # Separator line between MGUS and MM
    sep = len(mgus_sorted) - 0.5
    ax.axhline(sep, color="black", linewidth=2)

    # Labels
    ax.set_yticks([len(mgus_sorted) // 2, len(mgus_sorted) + len(mm_sorted) // 2])
    ax.set_yticklabels([f"MGUS (n={len(mgus_sorted)})",
                        f"MM (n={len(mm_sorted)})"], fontsize=12)

    # Chromosome arm labels on x-axis
    ax.set_xticks(range(len(arms)))
    # Simplify: show only chromosome numbers at boundaries
    xlabels = []
    for j, arm in enumerate(arms):
        chrom_num = arm.replace("chr", "").rstrip("pq")
        arm_type = arm[-1]
        if arm_type == "p":
            xlabels.append(chrom_num)
        else:
            xlabels.append("")
    ax.set_xticks(range(0, len(arms), 2))
    ax.set_xticklabels([arms[i].replace("chr", "").rstrip("pq")
                        for i in range(0, len(arms), 2)], fontsize=9)
    ax.set_xlabel("Chromosome", fontsize=12)

    # Add chromosome boundary lines
    for j in range(2, len(arms), 2):
        ax.axvline(j - 0.5, color="gray", linewidth=0.3, alpha=0.5)

    ax.set_title("CNA Landscape: MGUS vs MM (arm-level, sorted by CNA burden)",
                 fontsize=13)
    cbar = plt.colorbar(im, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Net CNA (amp - del fraction)", fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "06_cna_landscape.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved 06_cna_landscape.png ({len(mgus_sorted)} MGUS + {len(mm_sorted)} MM)")


# ============================================================================
# Enhancement 3: Cross-Dataset Validation
# ============================================================================

def cross_dataset_validation(merged_df, feature_cols):
    """Train on one dataset, test on the other."""
    print("\n--- Cross-Dataset Validation ---")

    y = (merged_df["label"] == "MM").astype(int).values
    X = merged_df[feature_cols].fillna(0).values
    datasets = merged_df["dataset"].values

    unique_ds = sorted(merged_df["dataset"].unique())
    if len(unique_ds) < 2:
        print("  Only one dataset found, skipping cross-dataset validation.")
        return

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Logistic Regression L1": LogisticRegression(
            C=1.0, solver="saga", max_iter=10000, l1_ratio=1.0,
            class_weight="balanced", random_state=RANDOM_STATE),
    }

    rows = []
    for train_ds, test_ds in [(unique_ds[0], unique_ds[1]),
                               (unique_ds[1], unique_ds[0])]:
        train_mask = datasets == train_ds
        test_mask = datasets == test_ds

        X_tr, y_tr = X[train_mask], y[train_mask]
        X_te, y_te = X[test_mask], y[test_mask]

        n_mgus_tr = (y_tr == 0).sum()
        n_mm_tr = (y_tr == 1).sum()
        n_mgus_te = (y_te == 0).sum()
        n_mm_te = (y_te == 1).sum()

        print(f"\n  Train: {train_ds} ({n_mgus_tr} MGUS + {n_mm_tr} MM) "
              f"-> Test: {test_ds} ({n_mgus_te} MGUS + {n_mm_te} MM)")

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        for name, model in models.items():
            model.fit(X_tr_s, y_tr)
            y_prob = model.predict_proba(X_te_s)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            auc = roc_auc_score(y_te, y_prob)
            f1 = f1_score(y_te, y_pred)
            bal_acc = balanced_accuracy_score(y_te, y_pred)
            sens = recall_score(y_te, y_pred, pos_label=1)
            spec = recall_score(y_te, y_pred, pos_label=0)

            row = {
                "train_dataset": train_ds, "test_dataset": test_ds,
                "model": name,
                "AUC": auc, "F1": f1, "BalAcc": bal_acc,
                "Sensitivity": sens, "Specificity": spec,
                "n_train": len(y_tr), "n_test": len(y_te),
            }
            rows.append(row)
            print(f"    {name}: AUC={auc:.3f}  F1={f1:.3f}  "
                  f"BalAcc={bal_acc:.3f}  Sens={sens:.3f}  Spec={spec:.3f}")

    result_df = pd.DataFrame(rows)
    result_df.to_csv(os.path.join(RESULTS_DIR, "cross_dataset_results.csv"), index=False)
    print(f"\n  Saved cross_dataset_results.csv")
    return result_df


# ============================================================================
# Enhancement 2: SHAP Analysis
# ============================================================================

def shap_analysis(merged_df, enhanced_cols):
    """SHAP analysis on HMM Enhanced RF model."""
    print("\n--- SHAP Analysis ---")

    try:
        import shap
    except ImportError:
        print("  SHAP not installed. Run: py -m pip install shap")
        print("  Skipping SHAP analysis.")
        return

    X = merged_df[enhanced_cols].fillna(0).values
    y = (merged_df["label"] == "MM").astype(int).values
    feature_names = enhanced_cols

    # Train RF on full dataset for explanation
    rf = RandomForestClassifier(
        n_estimators=500, max_depth=5, min_samples_leaf=3,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X, y)

    # SHAP TreeExplainer
    explainer = shap.TreeExplainer(rf)
    shap_values = explainer.shap_values(X)

    # For binary classification, use class 1 (MM) SHAP values
    if isinstance(shap_values, list):
        sv = shap_values[1]
    elif shap_values.ndim == 3:
        # Shape (n_samples, n_features, n_classes) — select class 1
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values

    # Plot 1: Beeswarm summary
    print("  Generating beeswarm plot...")
    fig, ax = plt.subplots(figsize=(12, 10))
    shap.summary_plot(sv, X, feature_names=feature_names, max_display=20,
                      show=False)
    plt.title("SHAP Feature Importance (HMM Enhanced RF)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "08_shap_beeswarm.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # Plot 2: Bar plot of mean |SHAP|
    print("  Generating SHAP bar plot...")
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(sv, X, feature_names=feature_names, plot_type="bar",
                      max_display=20, show=False)
    plt.title("Mean |SHAP Value| (HMM Enhanced RF)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "09_shap_bar.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    # Plot 3: Waterfall for interesting cases
    print("  Generating waterfall plots...")
    y_prob = rf.predict_proba(X)[:, 1]

    # Find a high-confidence correct MM prediction
    mm_mask = y == 1
    mm_probs = y_prob[mm_mask]
    mm_indices = np.where(mm_mask)[0]
    best_mm_idx = mm_indices[np.argmax(mm_probs)]

    # Find a borderline/misclassified MGUS case
    mgus_mask = y == 0
    mgus_probs = y_prob[mgus_mask]
    mgus_indices = np.where(mgus_mask)[0]
    # Pick the MGUS sample most likely to be classified as MM
    hardest_mgus_idx = mgus_indices[np.argmax(mgus_probs)]

    base_val = explainer.expected_value
    if isinstance(base_val, (list, np.ndarray)):
        base_val = base_val[1]

    # Waterfall plots for individual samples
    sample_id_mm = merged_df.index[best_mm_idx]
    sample_id_mgus = merged_df.index[hardest_mgus_idx]

    exp_mm = shap.Explanation(
        values=sv[best_mm_idx],
        base_values=base_val,
        data=X[best_mm_idx],
        feature_names=feature_names,
    )
    exp_mgus = shap.Explanation(
        values=sv[hardest_mgus_idx],
        base_values=base_val,
        data=X[hardest_mgus_idx],
        feature_names=feature_names,
    )

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
    ax2.set_title(f"Hardest MGUS ({sample_id_mgus}, P(MM)={y_prob[hardest_mgus_idx]:.3f}, "
                  f"pred {pred_label})", fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "10_shap_waterfall.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    print(f"  Saved 08_shap_beeswarm.png, 09_shap_bar.png, 10_shap_waterfall.png")


# ============================================================================
# Enhancement: ComBat Batch Correction
# ============================================================================

def combat_batch_correction(merged_df, feature_cols):
    """Apply ComBat batch correction using dataset as batch variable.
    Returns a corrected copy of merged_df."""
    print("\n--- ComBat Batch Correction ---")

    try:
        from combat.pycombat import pycombat
    except ImportError:
        print("  combat not installed. Run: py -m pip install combat")
        return None

    # pycombat expects features as rows, samples as columns
    data = merged_df[feature_cols].T
    batch = merged_df["dataset"].values

    print(f"  Input: {data.shape[1]} samples, {data.shape[0]} features")
    print(f"  Batches: {dict(pd.Series(batch).value_counts())}")

    corrected = pycombat(data, batch)

    # Build corrected dataframe
    corrected_df = merged_df.copy()
    corrected_df[feature_cols] = corrected.T.values

    print(f"  ComBat correction complete.")
    return corrected_df


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Post-hoc Analysis Suite")
    print("=" * 60)

    # Load data
    merged = pd.read_csv(os.path.join(FEAT_DIR, "feature_matrix_arm_all.csv"), index_col=0)
    merged = merged[merged["label"].isin(["MGUS", "MM"])]

    meta_cols = ["label", "dataset"]
    enhanced_cols = [c for c in merged.columns if c not in meta_cols]

    print(f"Loaded: {(merged['label']=='MGUS').sum()} MGUS + "
          f"{(merged['label']=='MM').sum()} MM = {len(merged)} samples")
    print(f"Features: {len(enhanced_cols)}")

    # Run all analyses
    plot_cna_landscape(merged)

    print("\n--- Cross-Dataset Validation (uncorrected) ---")
    cross_dataset_validation(merged, enhanced_cols)

    # ComBat batch correction
    corrected = combat_batch_correction(merged, enhanced_cols)
    if corrected is not None:
        # Save corrected features
        corrected.to_csv(os.path.join(FEAT_DIR, "feature_matrix_combat.csv"))
        print(f"  Saved feature_matrix_combat.csv")

        # Re-run cross-dataset with corrected features
        print("\n--- Cross-Dataset Validation (ComBat corrected) ---")
        cross_dataset_validation(corrected, enhanced_cols)

        # Also run standard CV on corrected features for comparison
        print("\n--- Standard CV on ComBat-corrected features ---")
        from sklearn.model_selection import RepeatedStratifiedKFold
        from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

        X_combat = corrected[enhanced_cols].fillna(0).values
        y = (corrected["label"] == "MM").astype(int).values

        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
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

    shap_analysis(merged, enhanced_cols)

    print("\n" + "=" * 60)
    print("  Analysis complete.")
    print("=" * 60)
