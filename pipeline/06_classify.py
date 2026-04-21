#!/usr/bin/env python3
"""Classification: HMM vs Raw features for MGUS/MM.

  - Fisher's exact test with BH correction per arm
  - RF + LR L1 with 5x10 repeated stratified CV
  - Paired statistical tests comparing HMM vs Raw
  - Plots: AUC, ROC, importance, metrics, confusion, CNA landscape, arm comparison
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
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
FEAT_DIR = os.path.join(BASE_DIR, "output", "features")
PLOT_DIR = os.path.join(BASE_DIR, "output", "plots")
RESULTS_DIR = os.path.join(BASE_DIR, "output", "results")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS, N_REPEATS, SEED = 5, 10, 42

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
ARM_ORDER = [f"chr{i}{a}" for i in range(1, 23) for a in ("p", "q")]


def make_models():
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Logistic Regression L1": LogisticRegression(
            C=1.0, solver="saga", max_iter=10000, l1_ratio=1.0,
            class_weight="balanced", random_state=SEED),
    }


def run_cv(X, y, feat_names, label):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    models = make_models()
    res = {n: {"auc": [], "f1": [], "bal_acc": [],
               "y_true": [], "y_prob": [], "y_pred": [], "imp": []}
           for n in models}

    for tr, te in cv.split(X, y):
        sc = StandardScaler()
        Xtr_s, Xte_s = sc.fit_transform(X[tr]), sc.transform(X[te])
        for name, mdl in models.items():
            if "Logistic" in name:
                mdl.fit(Xtr_s, y[tr]); prob = mdl.predict_proba(Xte_s)[:, 1]
            else:
                mdl.fit(X[tr], y[tr]); prob = mdl.predict_proba(X[te])[:, 1]
            pred = (prob >= 0.5).astype(int)
            res[name]["auc"].append(roc_auc_score(y[te], prob))
            res[name]["f1"].append(f1_score(y[te], pred))
            res[name]["bal_acc"].append(balanced_accuracy_score(y[te], pred))
            res[name]["y_true"].extend(y[te].tolist())
            res[name]["y_prob"].extend(prob.tolist())
            res[name]["y_pred"].extend(pred.tolist())
            if "Forest" in name:
                res[name]["imp"].append(mdl.feature_importances_)
            else:
                res[name]["imp"].append(np.abs(mdl.coef_[0]))

    rows = []
    print(f"\n  === {label} ===")
    for name in models:
        r = res[name]
        yt, yp = np.array(r["y_true"]), np.array(r["y_pred"])
        sens = recall_score(yt, yp, pos_label=1)
        spec = recall_score(yt, yp, pos_label=0)
        row = {"feature_set": label, "model": name,
               "AUC_mean": np.mean(r["auc"]), "AUC_std": np.std(r["auc"]),
               "F1_mean": np.mean(r["f1"]), "F1_std": np.std(r["f1"]),
               "BalAcc_mean": np.mean(r["bal_acc"]), "BalAcc_std": np.std(r["bal_acc"]),
               "Sensitivity": sens, "Specificity": spec}
        rows.append(row)
        print(f"  {name}: AUC={row['AUC_mean']:.3f}+/-{row['AUC_std']:.3f}  "
              f"F1={row['F1_mean']:.3f}  BalAcc={row['BalAcc_mean']:.3f}  "
              f"Sens={sens:.3f}  Spec={spec:.3f}")
        res[name]["mean_imp"] = np.mean(res[name]["imp"], axis=0)
    return res, rows


def region_tests(raw_df, raw_cols):
    """Per-arm Wilcoxon rank-sum tests on raw log2R.

    Tests whether mean log2 ratio per arm differs between MGUS and MM.
    More negative = deletion, more positive = amplification.
    BH correction for multiple comparisons.
    """
    print("\n" + "=" * 60)
    print("  PER-REGION TESTS (Wilcoxon rank-sum on raw log2R, BH corrected)")
    print("=" * 60)

    is_mm = (raw_df["label"] == "MM").values
    mean_cols = sorted(c for c in raw_cols if c.startswith("mean_"))
    arms = [c.replace("mean_", "") for c in mean_cols]

    rows = []
    for arm, col in zip(arms, mean_cols):
        mg_vals = raw_df.loc[~is_mm, col].values
        mm_vals = raw_df.loc[is_mm, col].values
        _, p = stats.mannwhitneyu(mm_vals, mg_vals, alternative="two-sided")

        mg_mean = float(mg_vals.mean())
        mm_mean = float(mm_vals.mean())
        diff = mm_mean - mg_mean

        rows.append({"arm": arm, "mgus_mean": mg_mean, "mm_mean": mm_mean,
                      "diff": diff, "p_value": p})

    df = pd.DataFrame(rows)
    reject, qvals, _, _ = multipletests(df["p_value"], method="fdr_bh", alpha=0.05)
    df["q_value"] = qvals
    df["significant"] = reject
    df = df.sort_values("q_value").reset_index(drop=True)

    sig = df[df["significant"]]
    print(f"\n  Significant (q<0.05): {len(sig)} / {len(df)}")
    if len(sig) > 0:
        print(f"  {'Arm':<8} {'MGUS mean':>10} {'MM mean':>10} {'Diff':>8} {'q-value':>10}")
        for _, r in sig.iterrows():
            print(f"  {r['arm']:<8} {r['mgus_mean']:>+10.4f} {r['mm_mean']:>+10.4f} "
                  f"{r['diff']:>+8.4f} {r['q_value']:>10.4e}")

    df.to_csv(os.path.join(RESULTS_DIR, "region_tests.csv"), index=False)
    return df


def stat_tests(results):
    """Paired t-test + Wilcoxon: HMM vs Raw."""
    print("\n  === HMM vs Raw (RF, paired across CV folds) ===")
    model = "Random Forest"
    rows = []
    for metric in ["auc", "f1", "bal_acc"]:
        a = np.array(results["HMM"][model][metric])
        b = np.array(results["Raw"][model][metric])
        diff = np.mean(a) - np.mean(b)
        t_stat, t_p = stats.ttest_rel(a, b)
        w_stat, w_p = stats.wilcoxon(a, b)
        sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
        winner = "HMM" if diff > 0 else "Raw"
        print(f"    {metric.upper()}: diff={diff:+.4f}  p={t_p:.4f} {sig}  ({winner})")
        rows.append({"metric": metric.upper(), "diff": diff,
                      "t_stat": t_stat, "t_p": t_p, "w_stat": w_stat, "w_p": w_p})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "statistical_tests.csv"), index=False)


if __name__ == "__main__":
    print("=" * 60)
    print("  Classification: HMM vs Raw")
    print("=" * 60)

    hmm_df = pd.read_csv(os.path.join(FEAT_DIR, "feature_matrix_hmm.csv"), index_col=0)
    raw_df = pd.read_csv(os.path.join(FEAT_DIR, "feature_matrix_raw.csv"), index_col=0)
    hmm_df = hmm_df[hmm_df["label"].isin(["MGUS", "MM"])]
    raw_df = raw_df.loc[hmm_df.index]
    y = (hmm_df["label"] == "MM").astype(int).values

    hmm_cols = [c for c in hmm_df.columns if c != "label"]
    raw_cols = [c for c in raw_df.columns if c != "label"]
    X_hmm = hmm_df[hmm_cols].fillna(0).values
    X_raw = raw_df[raw_cols].fillna(0).values

    n_mgus, n_mm = (y == 0).sum(), (y == 1).sum()
    print(f"  {n_mgus} MGUS + {n_mm} MM = {len(y)} samples")
    print(f"  HMM: {X_hmm.shape[1]} features  |  Raw: {X_raw.shape[1]} features")

    # Fisher's test
    # Region tests on Raw features (unbiased continuous signal)
    region_df = region_tests(raw_df, raw_cols)

    # Classification
    results = {}
    all_rows = []
    for name, X, cols in [("HMM", X_hmm, hmm_cols), ("Raw", X_raw, raw_cols)]:
        print(f"\nRunning CV: {name} ({X.shape[1]} features)...")
        res, rows = run_cv(X, y, cols, name)
        results[name] = res
        all_rows.extend(rows)
    summary_df = pd.DataFrame(all_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "classification_results.csv"), index=False)

    print("\n" + "=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print(summary_df[["feature_set", "model", "AUC_mean", "AUC_std",
                       "F1_mean", "BalAcc_mean", "Sensitivity", "Specificity"
                       ]].to_string(index=False))

    stat_tests(results)

    # ===== PLOTS =====
    print("\nGenerating plots...")
    feat_sets = summary_df["feature_set"].unique()
    model_names = summary_df["model"].unique()

    # 1: AUC bar
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(feat_sets)); w = 0.35
    for i, m in enumerate(model_names):
        sub = summary_df[summary_df["model"] == m]
        vals = [sub[sub["feature_set"] == f]["AUC_mean"].values[0] for f in feat_sets]
        errs = [sub[sub["feature_set"] == f]["AUC_std"].values[0] for f in feat_sets]
        ax.bar(x + i * w, vals, w, yerr=errs, label=m, capsize=4, alpha=0.85)
    ax.set_ylabel("AUC-ROC"); ax.set_title(f"HMM vs Raw ({len(y)} samples)")
    ax.set_xticks(x + w/2); ax.set_xticklabels(feat_sets)
    ax.legend(); ax.set_ylim(0.5, 1.0); ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "01_auc.png"), dpi=150); plt.close()

    # 2: ROC
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = {"HMM": "steelblue", "Raw": "firebrick"}
    for fs in feat_sets:
        for mn in model_names:
            r = results[fs][mn]
            fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
            auc = roc_auc_score(r["y_true"], r["y_prob"])
            ls = "-" if "Forest" in mn else "--"
            ms = "RF" if "Forest" in mn else "LR"
            ax.plot(fpr, tpr, color=colors[fs], ls=ls, lw=2,
                    label=f"{fs} {ms} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC: HMM vs Raw")
    ax.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "02_roc.png"), dpi=150); plt.close()

    # 3: Feature importance (HMM RF)
    imp = results["HMM"]["Random Forest"]["mean_imp"]
    top = np.argsort(imp)[-20:]
    fig, ax = plt.subplots(figsize=(10, 7))
    names = [hmm_cols[i] for i in top]
    clrs = ["steelblue" if n.startswith("del") or n.startswith("seg_mean_del") else
            "firebrick" if n.startswith("amp") or n.startswith("seg_mean_amp") else
            "goldenrod" for n in names]
    ax.barh(range(len(top)), imp[top], color=clrs, alpha=0.85)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Importance (RF)"); ax.set_title("Top 20 HMM Features")
    ax.legend(handles=[Line2D([0],[0],color="steelblue",lw=8,label="Del"),
                       Line2D([0],[0],color="firebrick",lw=8,label="Amp"),
                       Line2D([0],[0],color="goldenrod",lw=8,label="Derived")],
              loc="lower right")
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "03_importance.png"), dpi=150); plt.close()

    # 4: All metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (mc, sc, t) in zip(axes, [("AUC_mean","AUC_std","AUC-ROC"),
                                       ("F1_mean","F1_std","F1"),
                                       ("BalAcc_mean","BalAcc_std","Bal Acc")]):
        for i, m in enumerate(model_names):
            sub = summary_df[summary_df["model"] == m]
            vals = [sub[sub["feature_set"]==f][mc].values[0] for f in feat_sets]
            errs = [sub[sub["feature_set"]==f][sc].values[0] for f in feat_sets]
            ax.bar(x + i*w, vals, w, yerr=errs, label=m, capsize=4, alpha=0.85)
        ax.set_title(t); ax.set_xticks(x+w/2); ax.set_xticklabels(feat_sets)
        ax.set_ylim(0, 1); ax.axhline(0.5, color="gray", ls="--", lw=0.5)
        if ax == axes[0]: ax.legend(fontsize=8)
    plt.suptitle(f"All Metrics ({len(y)} samples)", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "04_metrics.png"), dpi=150,
                                     bbox_inches="tight"); plt.close()

    # 5: Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, fs in zip(axes, feat_sets):
        r = results[fs]["Random Forest"]
        cm = confusion_matrix(r["y_true"], r["y_pred"])
        cmn = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["MGUS","MM"]); ax.set_yticklabels(["MGUS","MM"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{fs} RF")
        for i in range(2):
            for j in range(2):
                c = "white" if cmn[i,j] > 0.5 else "black"
                ax.text(j, i, f"{cm[i,j]}\n({cmn[i,j]:.0%})", ha="center", va="center",
                        color=c, fontsize=10)
    plt.suptitle("Confusion Matrices (RF)"); plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "05_confusion.png"), dpi=150, bbox_inches="tight"); plt.close()

    # 6: CNA landscape heatmap
    del_cols_p = [f"del_{a}" for a in ARM_ORDER if f"del_{a}" in hmm_df.columns]
    amp_cols_p = [f"amp_{a}" for a in ARM_ORDER if f"amp_{a}" in hmm_df.columns]
    arms_p = [c.replace("del_", "") for c in del_cols_p]
    net = pd.DataFrame({a: hmm_df[f"amp_{a}"].fillna(0) - hmm_df[f"del_{a}"].fillna(0)
                        for a in arms_p}, index=hmm_df.index)
    burden = net.abs().mean(axis=1)
    mgus_s = burden.loc[hmm_df[hmm_df["label"]=="MGUS"].index].sort_values().index
    mm_s = burden.loc[hmm_df[hmm_df["label"]=="MM"].index].sort_values().index
    order = list(mgus_s) + list(mm_s)

    fig, ax = plt.subplots(figsize=(16, 8))
    im = ax.imshow(net.loc[order, arms_p].values, aspect="auto", cmap="RdBu_r",
                   vmin=-0.5, vmax=0.5, interpolation="nearest")
    ax.set_xticks(range(len(arms_p))); ax.set_xticklabels(arms_p, rotation=90, fontsize=6)
    ax.axhline(len(mgus_s) - 0.5, color="black", lw=2)
    ax.text(len(arms_p)+1, len(mgus_s)/2, "MGUS", va="center", fontsize=11, fontweight="bold")
    ax.text(len(arms_p)+1, len(mgus_s)+len(mm_s)/2, "MM", va="center", fontsize=11, fontweight="bold")
    ax.set_ylabel("Samples"); ax.set_title("CNA Landscape (blue=loss, red=gain)")
    plt.colorbar(im, ax=ax, shrink=0.6, pad=0.12)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "06_cna_landscape.png"), dpi=150,
                                     bbox_inches="tight"); plt.close()

    # 7: Per-arm comparison
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
    xa = np.arange(len(arms_p)); w2 = 0.35
    for ax, prefix, title in [(ax1, "del_", "Deletion"), (ax2, "amp_", "Amplification")]:
        cols = [f"{prefix}{a}" for a in arms_p]
        mg = hmm_df.loc[hmm_df["label"]=="MGUS", cols].mean().values
        mm = hmm_df.loc[hmm_df["label"]=="MM", cols].mean().values
        ax.bar(xa-w2/2, mg, w2, label="MGUS", color="steelblue", alpha=0.8)
        ax.bar(xa+w2/2, mm, w2, label="MM", color="firebrick", alpha=0.8)
        ax.set_ylabel(f"{title} fraction"); ax.set_title(f"{title} by Arm"); ax.legend()
    ax2.set_xticks(xa); ax2.set_xticklabels(arms_p, rotation=90, fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "07_arm_comparison.png"), dpi=150,
                                     bbox_inches="tight"); plt.close()

    print(f"\nPlots: {PLOT_DIR}")
    print(f"Results: {RESULTS_DIR}")
    print("Done.")
