###############################################################################
# Phase 8: Deep Learning (MLP) Classification
# Compares sklearn MLPClassifier against RF/LR baselines on all feature sets.
# Uses same 5×10 repeated stratified CV for fair comparison.
###############################################################################

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (roc_auc_score, f1_score, balanced_accuracy_score)
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEAT_DIR = os.path.join(BASE_DIR, "output", "features")
PLOT_DIR = os.path.join(BASE_DIR, "output", "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "output", "results")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS = 5
N_REPEATS = 10
RANDOM_STATE = 42


def run_dl_cv(X, y, feature_set_name):
    """Run 5×10 repeated stratified CV comparing MLP vs RF vs LR."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                  random_state=RANDOM_STATE)

    models = {
        "MLP": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu", solver="adam",
            alpha=0.01, batch_size=32,
            learning_rate="adaptive", learning_rate_init=0.001,
            max_iter=500, early_stopping=True,
            validation_fraction=0.15, n_iter_no_change=20,
            random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        "Logistic Regression L1": LogisticRegression(
            C=1.0, solver="saga", max_iter=10000, l1_ratio=1.0,
            class_weight="balanced", random_state=RANDOM_STATE),
    }

    results = {name: {"auc": [], "f1": [], "bal_acc": []} for name in models}

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        for name, model in models.items():
            # All models use scaled features for fair comparison
            model.fit(X_tr_s, y_tr)
            y_prob = model.predict_proba(X_te_s)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)

            results[name]["auc"].append(roc_auc_score(y_te, y_prob))
            results[name]["f1"].append(f1_score(y_te, y_pred))
            results[name]["bal_acc"].append(balanced_accuracy_score(y_te, y_pred))

    print(f"\n  === {feature_set_name} ===")
    summary_rows = []
    for name in models:
        r = results[name]
        row = {
            "model": name, "features": feature_set_name,
            "AUC_mean": np.mean(r["auc"]), "AUC_std": np.std(r["auc"]),
            "F1_mean": np.mean(r["f1"]), "F1_std": np.std(r["f1"]),
            "BalAcc_mean": np.mean(r["bal_acc"]), "BalAcc_std": np.std(r["bal_acc"]),
        }
        summary_rows.append(row)
        print(f"  {name}: AUC={row['AUC_mean']:.3f}±{row['AUC_std']:.3f}  "
              f"F1={row['F1_mean']:.3f}±{row['F1_std']:.3f}  "
              f"BalAcc={row['BalAcc_mean']:.3f}±{row['BalAcc_std']:.3f}")

    return results, summary_rows


if __name__ == "__main__":
    print("=" * 60)
    print("  Deep Learning: MLP vs RF vs LR")
    print("=" * 60)

    # Load features
    merged = pd.read_csv(os.path.join(FEAT_DIR, "feature_matrix_arm_all.csv"), index_col=0)
    merged = merged[merged["label"].isin(["MGUS", "MM"])]
    y = (merged["label"] == "MM").astype(int).values

    meta_cols = ["label", "dataset"]
    all_feat_cols = [c for c in merged.columns if c not in meta_cols]

    basic_cols = [c for c in all_feat_cols
                  if (c.startswith("del_") or c.startswith("amp_"))
                  and not c.startswith(("seg_mean_del_", "seg_mean_amp_"))]
    enhanced_cols = all_feat_cols

    X_basic = merged[basic_cols].fillna(0).values
    X_enhanced = merged[enhanced_cols].fillna(0).values

    # Load raw features (aligned to same arms as HMM)
    raw_path = os.path.join(FEAT_DIR, "feature_matrix_raw_arm.csv")
    if os.path.exists(raw_path):
        raw_df = pd.read_csv(raw_path, index_col=0)
        raw_df = raw_df.loc[merged.index]
        # Align to HMM arm set
        hmm_arms = sorted(set(c.replace("del_", "").replace("amp_", "")
                              for c in basic_cols))
        raw_aligned_cols = []
        for arm in hmm_arms:
            for prefix in ["mean_", "sd_"]:
                col = f"{prefix}{arm}"
                if col in raw_df.columns:
                    raw_aligned_cols.append(col)
        X_raw = raw_df[raw_aligned_cols].fillna(0).values
    else:
        print("WARNING: Raw features not found. Run 06_classify.py first.")
        X_raw = None

    # Combined
    X_combined = np.hstack([X_enhanced, X_raw]) if X_raw is not None else None

    print(f"Loaded: {(y==0).sum()} MGUS + {(y==1).sum()} MM = {len(y)} samples")

    # Run on all feature sets
    feature_sets = [
        ("HMM Basic", X_basic),
        ("HMM Enhanced", X_enhanced),
    ]
    if X_raw is not None:
        feature_sets.append(("Raw", X_raw))
    if X_combined is not None:
        feature_sets.append(("Combined", X_combined))

    all_results = {}
    all_summaries = []
    for name, X in feature_sets:
        print(f"\nRunning CV: {name} ({X.shape[1]} features)...")
        results, summaries = run_dl_cv(X, y, name)
        all_results[name] = results
        all_summaries.extend(summaries)

    # Save results
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "deep_learning_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    print(summary_df[["features", "model", "AUC_mean", "AUC_std",
                       "F1_mean", "BalAcc_mean"]].to_string(index=False))

    # Plot: MLP vs RF vs LR comparison on HMM Enhanced
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    metrics = [("AUC_mean", "AUC_std", "AUC-ROC"),
               ("F1_mean", "F1_std", "F1 Score"),
               ("BalAcc_mean", "BalAcc_std", "Balanced Accuracy")]

    # Focus on HMM Enhanced for the main comparison
    enhanced_rows = summary_df[summary_df["features"] == "HMM Enhanced"]
    model_names = enhanced_rows["model"].values
    bar_colors = ["darkorchid", "steelblue", "coral"]

    for ax, (mean_col, std_col, title) in zip(axes, metrics):
        means = enhanced_rows[mean_col].values
        stds = enhanced_rows[std_col].values
        bars = ax.bar(range(len(model_names)), means, yerr=stds,
                      capsize=5, alpha=0.85, color=bar_colors[:len(model_names)])
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(["MLP", "RF", "LR"], fontsize=10)
        ax.set_ylim(0.5, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
        for bar, m in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{m:.3f}", ha="center", fontsize=9)

    plt.suptitle(f"MLP vs RF vs LR on HMM Enhanced Features ({len(y)} samples)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "07_dl_comparison.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    print(f"\nPlot saved to {PLOT_DIR}/07_dl_comparison.png")
    print(f"Results saved to {RESULTS_DIR}/deep_learning_results.csv")
    print("Deep learning comparison complete.")
