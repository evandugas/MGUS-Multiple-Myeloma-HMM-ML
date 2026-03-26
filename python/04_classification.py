###############################################################################
# Phase 4: ML Classification
# Compares four feature sets for MGUS/MM classification:
#   1. HMM arm fractions only (baseline)
#   2. HMM enhanced (arm fractions + derived: burden, segments, means)
#   3. Raw LogRatio arm-level (mean + SD)
#   4. HMM + Raw combined
# Includes: confusion matrices, per-class metrics, statistical significance.
###############################################################################

import os
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

BASE_DIR = os.path.join("C:", os.sep, "Users", "Evan", "MGUS-Multiple-Myeloma-HMM-ML")
FEAT_DIR = os.path.join(BASE_DIR, "output", "features")
PLOT_DIR = os.path.join(BASE_DIR, "output", "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "output", "results")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42

CENTROMERES = {
    "chr1": 125e6, "chr2": 93.3e6, "chr3": 91e6, "chr4": 50.4e6,
    "chr5": 48.4e6, "chr6": 61e6, "chr7": 59.9e6, "chr8": 45.6e6,
    "chr9": 49e6, "chr10": 40.2e6, "chr11": 53.7e6, "chr12": 35.8e6,
    "chr13": 17.9e6, "chr14": 17.6e6, "chr15": 19e6, "chr16": 36.6e6,
    "chr17": 22.2e6, "chr18": 18.2e6, "chr19": 26.5e6, "chr20": 27.5e6,
    "chr21": 13.2e6, "chr22": 14.7e6,
}
CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]


def run_cv(X, y, feature_names, dataset_name):
    """Run repeated stratified k-fold CV with RF and LR."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                  random_state=RANDOM_STATE)

    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Logistic Regression L1": LogisticRegression(
            C=1.0, solver="saga", max_iter=10000, l1_ratio=1.0,
            class_weight="balanced", random_state=RANDOM_STATE),
    }

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
        print(f"  {name}: AUC={row['AUC_mean']:.3f}±{row['AUC_std']:.3f}  "
              f"F1={row['F1_mean']:.3f}±{row['F1_std']:.3f}  "
              f"BalAcc={row['BalAcc_mean']:.3f}±{row['BalAcc_std']:.3f}  "
              f"Sens={sensitivity:.3f}  Spec={specificity:.3f}")

    for name in models:
        imps = np.array(results[name]["importances"])
        results[name]["mean_importance"] = imps.mean(axis=0)

    return results, summary_rows


def statistical_tests(all_results):
    """Paired t-test and Wilcoxon on per-fold metrics between feature sets."""
    comparisons = [
        ("HMM Enhanced", "Raw Arm (mean+SD)"),
        ("HMM Enhanced", "HMM Arm Fractions"),
        ("HMM Enhanced", "HMM + Raw Combined"),
    ]
    model = "Random Forest"
    rows = []

    print("\n  === Statistical Significance (RF, paired across 50 CV folds) ===")
    for ds_a, ds_b in comparisons:
        if ds_a not in all_results or ds_b not in all_results:
            continue
        for metric in ["auc", "f1", "bal_acc"]:
            a = np.array(all_results[ds_a][model][metric])
            b = np.array(all_results[ds_b][model][metric])
            diff = np.mean(a) - np.mean(b)
            t_stat, t_pval = stats.ttest_rel(a, b)
            w_stat, w_pval = stats.wilcoxon(a, b)
            rows.append({
                "comparison": f"{ds_a} vs {ds_b}",
                "metric": metric.upper(),
                "mean_diff": diff,
                "t_statistic": t_stat, "t_pvalue": t_pval,
                "wilcoxon_statistic": w_stat, "wilcoxon_pvalue": w_pval,
            })
            sig = "***" if t_pval < 0.001 else "**" if t_pval < 0.01 else "*" if t_pval < 0.05 else "ns"
            print(f"  {ds_a} vs {ds_b} [{metric.upper()}]: "
                  f"diff={diff:+.4f}  p={t_pval:.4f} {sig}")

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
        # Normalize by row (true label)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["MGUS", "MM"], fontsize=10)
        ax.set_yticklabels(["MGUS", "MM"], fontsize=10)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)

        short = ds_name.split(" ")[0]
        if "Enhanced" in ds_name:
            short = "HMM+"
        elif "Combined" in ds_name:
            short = "Combined"
        ax.set_title(f"{short} RF", fontsize=12)

        for i in range(2):
            for j in range(2):
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i,j]}\n({cm_norm[i,j]:.1%})",
                        ha="center", va="center", color=color, fontsize=10)

    plt.suptitle("Confusion Matrices (RF, pooled across 50 CV folds)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "05_confusion_matrices.png"), dpi=150,
                bbox_inches="tight")
    plt.close()


def build_raw_arm_features(merged_df):
    """Build raw LogRatio arm-level features (mean + SD per arm) from the
    original probe data for each dataset's samples."""
    import gzip

    sample_ids = merged_df.index.tolist()
    labels = dict(zip(merged_df.index, merged_df["label"]))
    datasets = dict(zip(merged_df.index, merged_df["dataset"]))

    ds_samples = {}
    for sid in sample_ids:
        ds = datasets[sid]
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

    all_arms = sorted(set(a for sid in arm_stats for a in arm_stats[sid]))
    rows = []
    for sid in sample_ids:
        row = {"sample_id": sid, "label": labels[sid]}
        for arm_id in all_arms:
            s_val, sq, n = arm_stats[sid].get(arm_id, [0, 0, 0])
            mean = s_val / n if n > 0 else 0
            var = (sq / n - mean ** 2) if n > 1 else 0
            row[f"mean_{arm_id}"] = mean
            row[f"sd_{arm_id}"] = np.sqrt(max(var, 0))
        rows.append(row)

    result = pd.DataFrame(rows).set_index("sample_id")
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("  Classification: HMM vs Raw Features (196 samples)")
    print("=" * 60)

    # Load merged HMM features
    merged = pd.read_csv(os.path.join(FEAT_DIR, "feature_matrix_arm_all.csv"), index_col=0)
    merged = merged[merged["label"].isin(["MGUS", "MM"])]

    y = (merged["label"] == "MM").astype(int).values

    # Separate feature sets
    meta_cols = ["label", "dataset"]
    all_feat_cols = [c for c in merged.columns if c not in meta_cols]

    basic_cols = [c for c in all_feat_cols
                  if (c.startswith("del_") or c.startswith("amp_"))
                  and not c.startswith(("seg_mean_del_", "seg_mean_amp_"))]
    enhanced_cols = all_feat_cols

    X_basic = merged[basic_cols].fillna(0).values
    X_enhanced = merged[enhanced_cols].fillna(0).values

    print(f"Loaded: {(y==0).sum()} MGUS + {(y==1).sum()} MM = {len(y)} samples")
    print(f"HMM basic features: {X_basic.shape[1]}")
    print(f"HMM enhanced features: {X_enhanced.shape[1]}")

    # Build raw features
    print("\nBuilding raw LogRatio arm features...")
    raw_df = build_raw_arm_features(merged)
    raw_df.to_csv(os.path.join(FEAT_DIR, "feature_matrix_raw_arm.csv"))
    X_raw = raw_df.drop(columns=["label"]).fillna(0).values
    raw_feat_names = raw_df.drop(columns=["label"]).columns.tolist()
    print(f"Raw features: {X_raw.shape[1]}")

    # Enhancement 7: Combined HMM + Raw
    X_combined = np.hstack([X_enhanced, X_raw])
    combined_feat_names = enhanced_cols + raw_feat_names
    print(f"Combined features: {X_combined.shape[1]}")

    # Run CV on all four feature sets
    datasets = [
        ("HMM Arm Fractions", X_basic, basic_cols),
        ("HMM Enhanced", X_enhanced, enhanced_cols),
        ("Raw Arm (mean+SD)", X_raw, raw_feat_names),
        ("HMM + Raw Combined", X_combined, combined_feat_names),
    ]

    all_results = {}
    all_summaries = []

    for name, X, feat_names in datasets:
        print(f"\nRunning CV: {name} ({X.shape[1]} features)...")
        results, summaries = run_cv(X, y, feat_names, name)
        all_results[name] = results
        all_summaries.extend(summaries)

    # Save results
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "classification_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(summary_df[["dataset", "model", "AUC_mean", "AUC_std",
                       "F1_mean", "BalAcc_mean", "Sensitivity",
                       "Specificity"]].to_string(index=False))

    # Enhancement 6: Statistical significance
    stat_df = statistical_tests(all_results)

    # ---- Plots ---------------------------------------------------------------

    print("\nGenerating plots...")
    dataset_names = summary_df["dataset"].unique()
    model_names = summary_df["model"].unique()
    x = np.arange(len(dataset_names))
    width = 0.35
    colors = ["steelblue", "coral"]
    short_labels = []
    for d in dataset_names:
        if "Enhanced" in d:
            short_labels.append("HMM+")
        elif "Combined" in d:
            short_labels.append("Combined")
        elif "Raw" in d:
            short_labels.append("Raw")
        else:
            short_labels.append("HMM")

    # Plot 1: AUC comparison bar chart
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, model in enumerate(model_names):
        subset = summary_df[summary_df["model"] == model]
        means = [subset[subset["dataset"] == d]["AUC_mean"].values[0] for d in dataset_names]
        stds = [subset[subset["dataset"] == d]["AUC_std"].values[0] for d in dataset_names]
        ax.bar(x + i * width, means, width, yerr=stds, label=model,
               capsize=4, alpha=0.85, color=colors[i])
    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Classification: All Feature Sets (196 samples)", fontsize=13)
    ax.set_xticks(x + width / 2)
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
        ("HMM Arm Fractions", "Random Forest", "steelblue", "-"),
        ("HMM Arm Fractions", "Logistic Regression L1", "steelblue", "--"),
        ("HMM Enhanced", "Random Forest", "forestgreen", "-"),
        ("HMM Enhanced", "Logistic Regression L1", "forestgreen", "--"),
        ("Raw Arm (mean+SD)", "Random Forest", "firebrick", "-"),
        ("Raw Arm (mean+SD)", "Logistic Regression L1", "firebrick", "--"),
        ("HMM + Raw Combined", "Random Forest", "darkorange", "-"),
        ("HMM + Raw Combined", "Logistic Regression L1", "darkorange", "--"),
    ]

    for ds_name, model_name, color, ls in plot_configs:
        r = all_results[ds_name][model_name]
        y_true = np.array(r["y_true_all"])
        y_prob = np.array(r["y_prob_all"])
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        label_short = ds_name.split(" ")[0]
        if "Enhanced" in ds_name:
            label_short = "HMM+"
        elif "Combined" in ds_name:
            label_short = "Comb"
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
        else:
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
        Line2D([0], [0], color="goldenrod", lw=8, label="Derived (burden/segments)"),
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
            means = [subset[subset["dataset"] == d][mean_col].values[0] for d in dataset_names]
            stds = [subset[subset["dataset"] == d][std_col].values[0] for d in dataset_names]
            ax.bar(x + i * width, means, width, yerr=stds, label=model,
                   capsize=4, alpha=0.85, color=colors[i])
        ax.set_title(title, fontsize=12)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(short_labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
        if ax == axes[0]:
            ax.legend(fontsize=8)

    plt.suptitle("All Metrics: 4 Feature Sets (196 samples)", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "04_all_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Plot 5: Confusion matrices
    plot_confusion_matrices(all_results, dataset_names)

    print(f"\nPlots saved to {PLOT_DIR}")
    print(f"Results saved to {RESULTS_DIR}")
    print("Classification complete.")
