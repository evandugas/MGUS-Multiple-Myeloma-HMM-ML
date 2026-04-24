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
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             balanced_accuracy_score, roc_curve,
                             precision_recall_curve, confusion_matrix,
                             recall_score)
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
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 100

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
            C=1.0, solver="saga", max_iter=10000, penalty="l1",
            class_weight="balanced", random_state=SEED),
        "Logistic Regression ElasticNet": LogisticRegression(
            C=1.0, solver="saga", max_iter=10000,
            penalty="elasticnet", l1_ratio=0.5,
            class_weight="balanced", random_state=SEED),
    }


def run_cv(X, y, feat_names, label):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    models = make_models()
    res = {n: {"auc": [], "pr_auc": [], "f1": [], "bal_acc": [],
               "y_true": [], "y_prob": [], "y_pred": [], "imp": []}
           for n in models}

    for tr, te in cv.split(X, y):
        sc = StandardScaler()
        Xtr_s, Xte_s = sc.fit_transform(X[tr]), sc.transform(X[te])
        for name, mdl in models.items():
            if name == "Random Forest":
                mdl.fit(X[tr], y[tr]); prob = mdl.predict_proba(X[te])[:, 1]
            else:
                mdl.fit(Xtr_s, y[tr]); prob = mdl.predict_proba(Xte_s)[:, 1]
            pred = (prob >= 0.5).astype(int)
            res[name]["auc"].append(roc_auc_score(y[te], prob))
            res[name]["pr_auc"].append(average_precision_score(y[te], prob))
            res[name]["f1"].append(f1_score(y[te], pred))
            res[name]["bal_acc"].append(balanced_accuracy_score(y[te], pred))
            res[name]["y_true"].extend(y[te].tolist())
            res[name]["y_prob"].extend(prob.tolist())
            res[name]["y_pred"].extend(pred.tolist())
            if "Forest" in name:
                res[name]["imp"].append(mdl.feature_importances_)
            elif hasattr(mdl, "coef_"):
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
               "PR_AUC_mean": np.mean(r["pr_auc"]), "PR_AUC_std": np.std(r["pr_auc"]),
               "F1_mean": np.mean(r["f1"]), "F1_std": np.std(r["f1"]),
               "BalAcc_mean": np.mean(r["bal_acc"]), "BalAcc_std": np.std(r["bal_acc"]),
               "Sensitivity": sens, "Specificity": spec}
        rows.append(row)
        print(f"  {name}: AUC={row['AUC_mean']:.3f}+/-{row['AUC_std']:.3f}  "
              f"PR-AUC={row['PR_AUC_mean']:.3f}  "
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


def platform_confound_analysis(hmm_df, raw_df, y, platforms):
    """Re-run CV restricted to GPL11358 samples only.

    The original cohort is platform-confounded: all 90 MGUS are GPL11358, but
    MM spans 4 platforms (16 GPL11358 + 17 others). A classifier could learn
    "platform signature" rather than disease. Subsetting to GPL11358 (90 MGUS +
    16 MM) removes this confound. If AUC survives here, the HMM/Raw finding is
    real. If it collapses, platform was carrying the signal.
    """
    print("\n" + "=" * 60)
    print("  PLATFORM CONFOUND ANALYSIS (GPL11358 only)")
    print("=" * 60)

    is_gpl11358 = np.array([platforms.get(s, "") == "GPL11358" for s in hmm_df.index])
    n_mgus = int(((y == 0) & is_gpl11358).sum())
    n_mm = int(((y == 1) & is_gpl11358).sum())
    print(f"  Subset: {n_mgus} MGUS + {n_mm} MM  (dropped {int(~is_gpl11358).sum() if False else (~is_gpl11358).sum()} MM from other platforms)")

    hmm_cols = [c for c in hmm_df.columns if c != "label"]
    raw_cols = [c for c in raw_df.columns if c != "label"]
    X_hmm = hmm_df[hmm_cols].fillna(0).values[is_gpl11358]
    X_raw = raw_df[raw_cols].fillna(0).values[is_gpl11358]
    y_sub = y[is_gpl11358]

    rows = []
    for name, X in [("HMM", X_hmm), ("Raw", X_raw)]:
        print(f"\n  CV ({name}, {X.shape[1]} features, n={len(y_sub)}):")
        _, sub_rows = run_cv(X, y_sub, [c for c in (hmm_cols if name == "HMM" else raw_cols)], f"{name}_GPL11358")
        rows.extend(sub_rows)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "platform_confound.csv"), index=False)

    # Per-platform breakdown of MM samples (correctness of label prediction)
    # Use full-cohort RF predictions from results dict if available; otherwise skip.
    return df


def _run_cv_quiet(X, y, seed):
    """Minimal CV returning AUC per (model) averaged across folds. Used by permutation test."""
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=seed)
    aucs = {"Random Forest": [], "Logistic Regression L1": []}
    for tr, te in cv.split(X, y):
        sc = StandardScaler()
        Xtr_s, Xte_s = sc.fit_transform(X[tr]), sc.transform(X[te])
        rf = RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED, n_jobs=-1)
        rf.fit(X[tr], y[tr])
        aucs["Random Forest"].append(roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1]))
        lr = LogisticRegression(
            C=1.0, solver="saga", max_iter=10000, penalty="l1",
            class_weight="balanced", random_state=SEED)
        lr.fit(Xtr_s, y[tr])
        aucs["Logistic Regression L1"].append(roc_auc_score(y[te], lr.predict_proba(Xte_s)[:, 1]))
    return {m: float(np.mean(v)) for m, v in aucs.items()}


def permutation_test(X_hmm, X_raw, y, observed, n_perm=N_PERMUTATIONS):
    """Shuffle labels n_perm times, rerun full CV, build null AUC distribution.

    Empirical p = fraction of null AUCs >= observed AUC.
    Low p = pipeline is finding real signal, not artifacts / leakage.
    """
    print("\n" + "=" * 60)
    print(f"  LABEL PERMUTATION TEST ({n_perm} permutations)")
    print("=" * 60, flush=True)
    rng = np.random.RandomState(SEED)
    feat_sets = [("HMM", X_hmm), ("Raw", X_raw)]
    model_names = ["Random Forest", "Logistic Regression L1"]
    null = {fs: {m: [] for m in model_names} for fs, _ in feat_sets}

    for i in range(n_perm):
        y_perm = rng.permutation(y)
        for fs, X in feat_sets:
            aucs = _run_cv_quiet(X, y_perm, seed=SEED + i)
            for m in model_names:
                null[fs][m].append(aucs[m])
        if (i + 1) % 10 == 0:
            print(f"    {i + 1}/{n_perm} perms done", flush=True)

    print(f"\n  {'Feature':<6} {'Model':<25} {'Observed':>10} {'Null mean':>10} {'Null 95%':>18} {'p-value':>10}")
    rows = []
    for fs, _ in feat_sets:
        for m in model_names:
            obs = observed[fs][m]
            nulls = np.array(null[fs][m])
            p = float((nulls >= obs).sum() + 1) / (len(nulls) + 1)
            lo, hi = np.percentile(nulls, [2.5, 97.5])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"  {fs:<6} {m:<25} {obs:>10.3f} {np.mean(nulls):>10.3f} "
                  f"  [{lo:.3f}, {hi:.3f}] {p:>8.4f} {sig}")
            rows.append({"feature_set": fs, "model": m,
                         "observed_auc": obs,
                         "null_mean": float(np.mean(nulls)),
                         "null_lo": float(lo), "null_hi": float(hi),
                         "p_value": p, "n_perm": n_perm})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "permutation_test.csv"), index=False)

    # Plot null distributions
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (fs, m) in zip(axes.flat,
                           [("HMM", "Random Forest"), ("HMM", "Logistic Regression L1"),
                            ("Raw", "Random Forest"), ("Raw", "Logistic Regression L1")]):
        nulls = np.array(null[fs][m])
        ax.hist(nulls, bins=25, color="lightgray", edgecolor="black", alpha=0.8)
        obs = observed[fs][m]
        ax.axvline(obs, color="red", lw=2, label=f"Observed={obs:.3f}")
        ax.axvline(0.5, color="gray", ls="--", lw=0.8)
        ax.set_xlabel("AUC")
        ax.set_ylabel("Permutation count")
        p_row = [r for r in rows if r["feature_set"] == fs and r["model"] == m][0]
        ax.set_title(f"{fs} {m} (p={p_row['p_value']:.4f})")
        ax.legend()
    plt.suptitle(f"Label Permutation Test ({n_perm} perms)", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "08_permutation.png"), dpi=150,
                bbox_inches="tight")
    plt.close()
    return df


def operating_points(results):
    """Report metrics at multiple decision thresholds from pooled CV predictions.

    Thresholds:
      - 0.5 (default)
      - Youden's J maximum (best sens+spec - 1)
      - Minimum threshold achieving sensitivity >= 0.80 (clinical priority: catch MM)

    These use pooled test predictions from the existing CV (no re-training).
    """
    print("\n" + "=" * 60)
    print("  OPERATING POINT ANALYSIS (pooled CV predictions)")
    print("=" * 60)
    print(f"  {'Features':<4} {'Model':<24} {'Threshold rule':<22} "
          f"{'Thr':>6} {'Sens':>6} {'Spec':>6} {'F1':>6} {'BalAcc':>7}")
    rows = []

    # Grid sized for all (feat_set, model) combinations
    total = sum(len(md) for md in results.values())
    ncols = 3
    nrows = int(np.ceil(total / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).flatten()
    ax_i = 0

    for feat_set, model_dict in results.items():
        for model_name, r in model_dict.items():
            yt = np.array(r["y_true"])
            yp = np.array(r["y_prob"])
            fpr, tpr, thr_roc = roc_curve(yt, yp)
            thr_roc = np.clip(thr_roc, 0, 1)

            # Youden's J
            j = tpr - fpr
            j_idx = int(np.argmax(j))
            thr_youden = float(thr_roc[j_idx])

            # Highest threshold where sensitivity >= 0.80 (catches >=80% of MM)
            # roc_curve returns thresholds in DECREASING order; tpr increases from 0 to 1.
            # First index where tpr crosses 0.80 corresponds to the highest threshold
            # still achieving that sensitivity.
            sens80_mask = tpr >= 0.80
            if sens80_mask.any():
                idx80 = int(np.min(np.where(sens80_mask)))
                thr_sens80 = float(thr_roc[idx80])
            else:
                thr_sens80 = 0.0

            for thr_name, thr in [("default (0.5)", 0.5),
                                   ("Youden's J max", thr_youden),
                                   ("Sens >= 0.80", thr_sens80)]:
                pred = (yp >= thr).astype(int)
                sens = recall_score(yt, pred, pos_label=1) if pred.sum() > 0 else 0.0
                spec = recall_score(yt, pred, pos_label=0)
                f1 = f1_score(yt, pred) if pred.sum() > 0 else 0.0
                bacc = balanced_accuracy_score(yt, pred)
                rows.append({"feature_set": feat_set, "model": model_name,
                              "threshold_rule": thr_name, "threshold": thr,
                              "sens": sens, "spec": spec, "F1": f1, "BalAcc": bacc})
                print(f"  {feat_set:<4} {model_name:<24} {thr_name:<22} "
                      f"{thr:>6.3f} {sens:>6.3f} {spec:>6.3f} {f1:>6.3f} {bacc:>7.3f}")

            # Plot sens/spec curves
            ax = axes[ax_i]
            thresholds = np.linspace(0.01, 0.99, 99)
            s_list, p_list = [], []
            for t in thresholds:
                pr = (yp >= t).astype(int)
                if pr.sum() == 0:
                    s_list.append(0.0); p_list.append(1.0); continue
                s_list.append(recall_score(yt, pr, pos_label=1))
                p_list.append(recall_score(yt, pr, pos_label=0))
            ax.plot(thresholds, s_list, label="Sensitivity", color="firebrick", lw=2)
            ax.plot(thresholds, p_list, label="Specificity", color="steelblue", lw=2)
            ax.axvline(0.5, color="gray", ls=":", lw=0.8, alpha=0.6)
            ax.axvline(thr_youden, color="goldenrod", ls="--", lw=1, label=f"Youden={thr_youden:.2f}")
            ax.axvline(thr_sens80, color="darkgreen", ls="--", lw=1, label=f"Sens>=0.8 @ {thr_sens80:.2f}")
            ax.axhline(0.80, color="firebrick", ls=":", lw=0.5, alpha=0.5)
            ax.set_title(f"{feat_set} {model_name}")
            ax.set_xlabel("Decision threshold"); ax.set_ylabel("Metric")
            ax.legend(fontsize=8, loc="center right")
            ax.set_ylim(0, 1.02); ax.grid(alpha=0.3)
            ax_i += 1

    # Hide unused subplots
    for k in range(ax_i, len(axes)):
        axes[k].set_visible(False)

    plt.suptitle("Sensitivity/Specificity vs Threshold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "09_operating_points.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "operating_points.csv"), index=False)
    return df


def bootstrap_cis(results, n_boot=N_BOOTSTRAP):
    """Bootstrap 95% CIs on pooled CV predictions for AUC/F1/BalAcc.

    Resamples (y_true, y_prob) pairs with replacement. More robust than
    fold-std because it accounts for test-set overlap across repeated folds.
    """
    print("\n" + "=" * 60)
    print(f"  BOOTSTRAP 95% CIs ({n_boot} resamples)")
    print("=" * 60)
    rng = np.random.RandomState(SEED)
    rows = []
    for feat_set, model_dict in results.items():
        for model_name, r in model_dict.items():
            yt = np.array(r["y_true"])
            yp = np.array(r["y_prob"])
            yd = np.array(r["y_pred"])
            n = len(yt)
            aucs, pr_aucs, f1s, bas = [], [], [], []
            for _ in range(n_boot):
                idx = rng.randint(0, n, size=n)
                if len(np.unique(yt[idx])) < 2:
                    continue
                aucs.append(roc_auc_score(yt[idx], yp[idx]))
                pr_aucs.append(average_precision_score(yt[idx], yp[idx]))
                f1s.append(f1_score(yt[idx], yd[idx]))
                bas.append(balanced_accuracy_score(yt[idx], yd[idx]))
            row = {"feature_set": feat_set, "model": model_name}
            for metric, vals in [("AUC", aucs), ("PR_AUC", pr_aucs), ("F1", f1s), ("BalAcc", bas)]:
                if vals:
                    lo, hi = np.percentile(vals, [2.5, 97.5])
                    row[f"{metric}_lo"] = lo
                    row[f"{metric}_hi"] = hi
                    row[f"{metric}_mean"] = float(np.mean(vals))
            rows.append(row)
            print(f"  {feat_set:<4} {model_name:<24} "
                  f"AUC {row['AUC_mean']:.3f} [{row['AUC_lo']:.3f}, {row['AUC_hi']:.3f}]  "
                  f"PR {row['PR_AUC_mean']:.3f} [{row['PR_AUC_lo']:.3f}, {row['PR_AUC_hi']:.3f}]  "
                  f"F1 {row['F1_mean']:.3f} [{row['F1_lo']:.3f}, {row['F1_hi']:.3f}]  "
                  f"BalAcc {row['BalAcc_mean']:.3f} [{row['BalAcc_lo']:.3f}, {row['BalAcc_hi']:.3f}]")
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "bootstrap_cis.csv"), index=False)
    return df


def stat_tests(results):
    """Paired t-test + Wilcoxon: HMM vs Raw and HMM80 vs Raw, all models."""
    print("\n  === Paired tests across CV folds ===")
    rows = []
    feat_pairs = []
    if "HMM" in results and "Raw" in results:
        feat_pairs.append(("HMM", "Raw"))
    if "HMM80" in results and "Raw" in results:
        feat_pairs.append(("HMM80", "Raw"))

    for a_name, b_name in feat_pairs:
        for model in results[a_name]:
            if model not in results[b_name]:
                continue
            for metric in ["auc", "pr_auc", "f1", "bal_acc"]:
                a = np.array(results[a_name][model][metric])
                b = np.array(results[b_name][model][metric])
                diff = float(np.mean(a) - np.mean(b))
                t_stat, t_p = stats.ttest_rel(a, b)
                try:
                    w_stat, w_p = stats.wilcoxon(a, b)
                except ValueError:
                    w_stat, w_p = np.nan, np.nan
                sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
                winner = a_name if diff > 0 else b_name
                print(f"    {a_name:<6} vs {b_name:<4} {model:<24} "
                      f"{metric.upper():<7} diff={diff:+.4f}  p={t_p:.4f} {sig}  ({winner})")
                rows.append({"comparison": f"{a_name}_vs_{b_name}", "model": model,
                              "metric": metric.upper(), "diff": diff,
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

    # Platform metadata (for confound analysis)
    labels_csv = os.path.join(BASE_DIR, "data", "GSE77975", "sample_labels.csv")
    platforms = {}
    if os.path.exists(labels_csv):
        meta = pd.read_csv(labels_csv)
        if "platform" in meta.columns:
            platforms = dict(zip(meta["sample_id"], meta["platform"]))

    hmm_cols = [c for c in hmm_df.columns if c != "label"]
    raw_cols = [c for c in raw_df.columns if c != "label"]
    X_hmm = hmm_df[hmm_cols].fillna(0).values
    X_raw = raw_df[raw_cols].fillna(0).values

    # HMM-80: matched feature count with Raw (del + amp fractions only, 40 arms x 2 = 80)
    hmm80_cols = [c for c in hmm_cols
                   if c.startswith("del_") or c.startswith("amp_")]
    X_hmm80 = hmm_df[hmm80_cols].fillna(0).values

    n_mgus, n_mm = (y == 0).sum(), (y == 1).sum()
    print(f"  {n_mgus} MGUS + {n_mm} MM = {len(y)} samples")
    print(f"  HMM: {X_hmm.shape[1]} features  |  HMM-80: {X_hmm80.shape[1]}  "
          f"|  Raw: {X_raw.shape[1]} features")

    # Fisher's test
    # Region tests on Raw features (unbiased continuous signal)
    region_df = region_tests(raw_df, raw_cols)

    # Classification
    results = {}
    all_rows = []
    for name, X, cols in [("HMM", X_hmm, hmm_cols),
                           ("HMM80", X_hmm80, hmm80_cols),
                           ("Raw", X_raw, raw_cols)]:
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
    operating_points(results)
    ci_df = bootstrap_cis(results)

    # Platform confound analysis (only if platform metadata available)
    if platforms:
        platform_confound_analysis(hmm_df, raw_df, y, platforms)

    # Permutation test (slow: ~100 perms x 2 feature sets x 2 models x 50-fold CV)
    observed_auc = {fs: {m: float(np.mean(results[fs][m]["auc"]))
                          for m in results[fs]} for fs in results}
    permutation_test(X_hmm, X_raw, y, observed_auc)

    # ===== PLOTS =====
    print("\nGenerating plots...")
    feat_sets = summary_df["feature_set"].unique()
    model_names = summary_df["model"].unique()

    # 1: AUC bar
    n_models = len(model_names)
    w = 0.8 / n_models
    fig, ax = plt.subplots(figsize=(max(8, 2 * len(feat_sets) + 2), 5))
    x = np.arange(len(feat_sets))
    for i, m in enumerate(model_names):
        sub = summary_df[summary_df["model"] == m]
        vals = [sub[sub["feature_set"] == f]["AUC_mean"].values[0] for f in feat_sets]
        errs = [sub[sub["feature_set"] == f]["AUC_std"].values[0] for f in feat_sets]
        ax.bar(x + i * w, vals, w, yerr=errs, label=m, capsize=4, alpha=0.85)
    ax.set_ylabel("AUC-ROC"); ax.set_title(f"HMM vs Raw ({len(y)} samples)")
    ax.set_xticks(x + w * (n_models - 1) / 2); ax.set_xticklabels(feat_sets)
    ax.legend(fontsize=9); ax.set_ylim(0.5, 1.0); ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "01_auc.png"), dpi=150); plt.close()

    # 2: ROC
    fig, ax = plt.subplots(figsize=(9, 9))
    colors = {"HMM": "steelblue", "HMM80": "goldenrod", "Raw": "firebrick"}
    linestyles = {"Random Forest": "-",
                   "Logistic Regression L1": "--",
                   "Logistic Regression ElasticNet": ":"}
    for fs in feat_sets:
        for mn in model_names:
            r = results[fs][mn]
            fpr, tpr, _ = roc_curve(r["y_true"], r["y_prob"])
            auc = roc_auc_score(r["y_true"], r["y_prob"])
            ls = linestyles.get(mn, "-")
            ms = "RF" if "Forest" in mn else ("LR-L1" if "L1" in mn else "LR-EN")
            ax.plot(fpr, tpr, color=colors.get(fs, "gray"), ls=ls, lw=2,
                    label=f"{fs} {ms} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.set_title("ROC curves")
    ax.legend(loc="lower right", fontsize=8); plt.tight_layout()
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

    # 4: All metrics (now 4 panels: AUC, PR-AUC, F1, BalAcc)
    fig, axes = plt.subplots(1, 4, figsize=(5 * len(feat_sets), 5))
    for ax, (mc, sc, t) in zip(axes, [("AUC_mean","AUC_std","AUC-ROC"),
                                       ("PR_AUC_mean","PR_AUC_std","PR-AUC"),
                                       ("F1_mean","F1_std","F1"),
                                       ("BalAcc_mean","BalAcc_std","Bal Acc")]):
        for i, m in enumerate(model_names):
            sub = summary_df[summary_df["model"] == m]
            vals = [sub[sub["feature_set"]==f][mc].values[0] for f in feat_sets]
            errs = [sub[sub["feature_set"]==f][sc].values[0] for f in feat_sets]
            ax.bar(x + i*w, vals, w, yerr=errs, label=m, capsize=4, alpha=0.85)
        ax.set_title(t); ax.set_xticks(x + w * (n_models - 1) / 2)
        ax.set_xticklabels(feat_sets, fontsize=9)
        ax.set_ylim(0, 1); ax.axhline(0.5, color="gray", ls="--", lw=0.5)
        if ax == axes[0]: ax.legend(fontsize=8)
    plt.suptitle(f"All Metrics ({len(y)} samples)", y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(PLOT_DIR, "04_metrics.png"), dpi=150,
                                     bbox_inches="tight"); plt.close()

    # 5: Confusion matrices
    fig, axes = plt.subplots(1, len(feat_sets), figsize=(4 * len(feat_sets), 4))
    if len(feat_sets) == 1:
        axes = [axes]
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
