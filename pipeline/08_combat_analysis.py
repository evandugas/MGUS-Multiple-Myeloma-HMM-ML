#!/usr/bin/env python3
"""ComBat batch correction sensitivity analysis.

The cohort is platform-confounded: all 90 MGUS samples come from GPL11358, but
MM samples are spread across 4 platforms. ComBat adjusts per-feature batch
effects while preserving the disease label signal (as a protected covariate).

We apply ComBat to the Raw arm-level features only (continuous, approximately
normal). HMM fractions are bounded [0,1] and don't satisfy ComBat assumptions.

Interpretation:
  - If classification AUC stays high after ComBat -> disease signal survives
    beyond platform effects. Supports the "real signal" interpretation.
  - If AUC collapses -> platform was carrying most of the apparent signal.

Note: ComBat cannot perfectly untangle confounded effects (all MGUS on one
platform means the disease and platform variances are partly inseparable).
This is a sensitivity analysis, not a fix.
"""

import os, sys, csv, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:
    from combat.pycombat import pycombat
except ImportError:
    print("combat-pycombat not installed. pip install combat-pycombat")
    sys.exit(1)

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             balanced_accuracy_score, recall_score)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "GSE77975")
FEAT_DIR = os.path.join(BASE_DIR, "output", "features")
RESULTS_DIR = os.path.join(BASE_DIR, "output", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SPLITS, N_REPEATS, SEED = 5, 10, 42


def run_cv(X, y, label):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=SEED)
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=500, max_depth=5, min_samples_leaf=3,
            class_weight="balanced", random_state=SEED, n_jobs=-1),
        "Logistic Regression L1": LogisticRegression(
            C=1.0, solver="saga", max_iter=10000, penalty="l1",
            class_weight="balanced", random_state=SEED),
    }
    res = {n: {"auc": [], "pr_auc": [], "f1": [], "bal_acc": [],
               "y_true": [], "y_pred": []}
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
            res[name]["y_pred"].extend(pred.tolist())

    rows = []
    print(f"\n  === {label} ===")
    for name, r in res.items():
        yt, yp = np.array(r["y_true"]), np.array(r["y_pred"])
        sens = recall_score(yt, yp, pos_label=1)
        spec = recall_score(yt, yp, pos_label=0)
        row = {"feature_set": label, "model": name,
                "AUC_mean": float(np.mean(r["auc"])),
                "AUC_std":  float(np.std(r["auc"])),
                "PR_AUC_mean": float(np.mean(r["pr_auc"])),
                "F1_mean":  float(np.mean(r["f1"])),
                "BalAcc_mean": float(np.mean(r["bal_acc"])),
                "Sensitivity": sens, "Specificity": spec}
        rows.append(row)
        print(f"  {name}: AUC={row['AUC_mean']:.3f}+/-{row['AUC_std']:.3f}  "
              f"PR-AUC={row['PR_AUC_mean']:.3f}  "
              f"F1={row['F1_mean']:.3f}  BalAcc={row['BalAcc_mean']:.3f}  "
              f"Sens={sens:.3f}  Spec={spec:.3f}")
    return rows


if __name__ == "__main__":
    print("=" * 60)
    print("  ComBat batch correction sensitivity analysis")
    print("=" * 60)

    # Load Raw features + platform metadata
    raw_df = pd.read_csv(os.path.join(FEAT_DIR, "feature_matrix_raw.csv"),
                         index_col=0)
    raw_df = raw_df[raw_df["label"].isin(["MGUS", "MM"])]

    meta = pd.read_csv(os.path.join(DATA_DIR, "sample_labels.csv"))
    platforms = dict(zip(meta["sample_id"], meta["platform"]))

    sample_ids = list(raw_df.index)
    y = (raw_df["label"] == "MM").astype(int).values
    batch = np.array([platforms.get(s, "unknown") for s in sample_ids])

    feature_cols = [c for c in raw_df.columns if c != "label"]
    X = raw_df[feature_cols].fillna(0).values

    print(f"  {len(sample_ids)} samples | {len(feature_cols)} Raw features")
    print(f"  Platforms: {dict(zip(*np.unique(batch, return_counts=True)))}")

    # ComBat expects features x samples (rows = features, columns = samples)
    df_for_combat = pd.DataFrame(X.T, index=feature_cols, columns=sample_ids)

    # Batch as pandas Series, aligned with sample_ids
    batch_series = pd.Series(batch, index=sample_ids)

    # pycombat's `mod=` parameter (protect a covariate) is broken in the installed
    # version — fails at internal `mod == []` check regardless of input type.
    # So we run ComBat WITHOUT disease protection. In a perfectly confounded
    # design (all 90 MGUS on GPL11358), removing batch variance also removes
    # most disease variance; this shows how much of the Raw+LR-L1 performance
    # was platform-driven vs disease-driven.
    print("\n  Running ComBat (no covariate protection, confounded design)...",
          flush=True)
    adj = pycombat(df_for_combat, batch_series)

    X_adj = adj.values.T  # back to samples x features
    print(f"  Adjusted matrix: {X_adj.shape}")

    # Classification: un-adjusted vs ComBat-adjusted
    rows_unadj = run_cv(X, y, "Raw (uncorrected)")
    rows_adj = run_cv(X_adj, y, "Raw (ComBat-adjusted)")

    df = pd.DataFrame(rows_unadj + rows_adj)
    out_path = os.path.join(RESULTS_DIR, "combat_analysis.csv")
    df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")
    print("  Done.")
