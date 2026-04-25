#!/usr/bin/env python3
"""Leakage-free classification: retrain HMM inside each CV fold.

The main pipeline (05 + 06) trains the cohort HMM on all 123 samples before
CV-splitting — so test-sample probes inform the emission parameters. This is
mild but real leakage. Here, for each fold we:

  1. Train cohort HMM on training samples ONLY
  2. Decode all 123 samples with fold-specific HMM
  3. Extract arm-level HMM features
  4. Fit RF + LR-L1 on training features; evaluate on test features

Raw features have no leakage (per-sample stats), so they are computed once.

Runtime: ~50 folds * (HMM train + 123 Viterbi decodes) ~ 1-2 hours.
"""

import os, sys, gc, warnings, csv, logging, pickle, importlib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             f1_score, balanced_accuracy_score,
                             precision_recall_curve, recall_score)
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.getLogger("hmmlearn").setLevel(logging.ERROR)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "GSE77975")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FEAT_DIR = os.path.join(OUTPUT_DIR, "features")
PROBES_DIR = os.path.join(OUTPUT_DIR, "probes")
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

sys.path.insert(0, os.path.join(BASE_DIR, "pipeline"))
_hmm = importlib.import_module("04_hmm_core")
_proc = importlib.import_module("05_process")

CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]
N_STATES = 5   # fallback if PER_FOLD_BIC is False
PER_FOLD_BIC = True   # redo BIC state selection inside each CV fold (no leakage)
N_SPLITS, N_REPEATS, SEED = 5, 10, 42
N_WORKERS = min(8, os.cpu_count() or 4)


def _decode_one(args):
    sid, chrom_data, n_states, means, covars, transmat = args
    warnings.filterwarnings("ignore")
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)
    return _hmm.decode_sample(chrom_data, n_states, means, covars, transmat, sid)


def load_probes():
    """Load cleaned probe dataframe saved by 05_process.py."""
    pq = os.path.join(PROBES_DIR, "probes_clean.parquet")
    pk = os.path.join(PROBES_DIR, "probes_clean.pkl")
    if os.path.exists(pq):
        return pd.read_parquet(pq)
    if os.path.exists(pk):
        return pd.read_pickle(pk)
    raise FileNotFoundError(
        f"Neither {pq} nor {pk} exists. Run 05_process.py first.")


def build_sample_chrom(probe_df):
    """Reorganize probes into dict[sample_id] -> dict[chrom] -> (lr, starts)."""
    sample_chrom = {}
    for sid, sdf in probe_df.groupby("sample_id"):
        cd = {}
        for chrom in CHROMOSOMES:
            mask = sdf["chr"] == chrom
            if mask.sum() < 20:
                continue
            cdf = sdf[mask].sort_values("start")
            cd[chrom] = (cdf["LogRatio"].values, cdf["start"].values)
        sample_chrom[sid] = cd
    return sample_chrom


def fold_hmm_features(train_sids, all_sids, sample_chrom, labels_dict,
                       per_fold_bic=PER_FOLD_BIC):
    """Train HMM on train_sids only, decode all samples, return feature matrix.

    If per_fold_bic=True, redo BIC state selection (3 vs 4 vs 5) on training
    sequences only before training the model. Otherwise uses N_STATES fallback.
    """
    train_seqs = []
    for sid in train_sids:
        for chrom in sample_chrom[sid]:
            train_seqs.append(sample_chrom[sid][chrom][0])

    if per_fold_bic:
        n_states = _hmm.select_n_states(train_seqs)
    else:
        n_states = N_STATES

    n_states, means, covars, transmat = _hmm.train_cohort_model(
        train_seqs, n_states)

    decode_args = [(sid, sample_chrom[sid], n_states, means, covars, transmat)
                   for sid in all_sids]
    all_states = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futs = [ex.submit(_decode_one, a) for a in decode_args]
        for f in as_completed(futs):
            all_states.extend(f.result())

    states_df = pd.DataFrame(all_states,
                             columns=["sample_id", "chr", "start", "log_ratio", "state"])
    feat = _proc.compute_hmm_features(states_df, labels_dict)
    return feat, n_states


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


def evaluate(X_tr, X_te, y_tr, y_te, res):
    """Train both models on X_tr/y_tr, evaluate on X_te/y_te, append to res."""
    sc = StandardScaler()
    Xtr_s, Xte_s = sc.fit_transform(X_tr), sc.transform(X_te)
    models = make_models()
    for name, mdl in models.items():
        if name == "Random Forest":
            mdl.fit(X_tr, y_tr); prob = mdl.predict_proba(X_te)[:, 1]
        else:
            mdl.fit(Xtr_s, y_tr); prob = mdl.predict_proba(Xte_s)[:, 1]
        pred = (prob >= 0.5).astype(int)
        res[name]["auc"].append(roc_auc_score(y_te, prob))
        res[name]["pr_auc"].append(average_precision_score(y_te, prob))
        res[name]["f1"].append(f1_score(y_te, pred))
        res[name]["bal_acc"].append(balanced_accuracy_score(y_te, pred))
        res[name]["y_true"].extend(y_te.tolist())
        res[name]["y_prob"].extend(prob.tolist())
        res[name]["y_pred"].extend(pred.tolist())


def summarize(res, label):
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
               "PR_AUC_std":  float(np.std(r["pr_auc"])),
               "F1_mean":  float(np.mean(r["f1"])),
               "F1_std":   float(np.std(r["f1"])),
               "BalAcc_mean": float(np.mean(r["bal_acc"])),
               "BalAcc_std":  float(np.std(r["bal_acc"])),
               "Sensitivity": sens, "Specificity": spec}
        rows.append(row)
        print(f"  {name}: AUC={row['AUC_mean']:.3f}+/-{row['AUC_std']:.3f}  "
              f"PR-AUC={row['PR_AUC_mean']:.3f}  "
              f"F1={row['F1_mean']:.3f}  BalAcc={row['BalAcc_mean']:.3f}  "
              f"Sens={sens:.3f}  Spec={spec:.3f}")
    return rows


def align_features(feat_df, sample_ids, feature_cols=None):
    """Reindex feature matrix to sample_ids order; align column set if provided."""
    feat_df = feat_df.reindex(sample_ids)
    cols = [c for c in feat_df.columns if c != "label"]
    if feature_cols is not None:
        # Ensure same column set as first fold (fill missing with 0)
        for c in feature_cols:
            if c not in feat_df.columns:
                feat_df[c] = 0
        cols = feature_cols
    X = feat_df[cols].fillna(0).values
    return X, cols


if __name__ == "__main__":
    print("=" * 60)
    print("  LEAKAGE-FREE Classification: fold-wise HMM retraining")
    print("=" * 60)

    # Labels
    labels_dict = {}
    with open(os.path.join(DATA_DIR, "sample_labels.csv")) as f:
        for row in csv.DictReader(f):
            labels_dict[row["sample_id"]] = row["label"]

    # Probes
    print("  Loading cleaned probes...", flush=True)
    probe_df = load_probes()
    probe_df = probe_df[probe_df["sample_id"].isin(labels_dict)]
    print(f"  {len(probe_df):,} rows, {probe_df['sample_id'].nunique()} samples",
          flush=True)

    sample_chrom = build_sample_chrom(probe_df)

    # Raw features: no leakage, compute once
    print("  Computing Raw features (once, no leakage)...", flush=True)
    _, binned_df = _proc.build_binned_sample_chrom(probe_df)
    raw_feat = _proc.compute_raw_features_from_bins(binned_df, labels_dict)
    raw_feat = raw_feat[raw_feat["label"].isin(["MGUS", "MM"])]
    sample_ids = list(raw_feat.index)
    y = (raw_feat["label"] == "MM").astype(int).values
    X_raw_full, raw_cols = align_features(raw_feat, sample_ids)
    print(f"  {len(sample_ids)} samples | Raw features: {len(raw_cols)}",
          flush=True)

    # Per-fold HMM
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS,
                                 random_state=SEED)
    hmm_res = {m: {"auc": [], "pr_auc": [], "f1": [], "bal_acc": [],
                    "y_true": [], "y_prob": [], "y_pred": []}
                for m in ["Random Forest", "Logistic Regression L1", "Logistic Regression ElasticNet"]}
    raw_res = {m: {"auc": [], "pr_auc": [], "f1": [], "bal_acc": [],
                    "y_true": [], "y_prob": [], "y_pred": []}
                for m in ["Random Forest", "Logistic Regression L1", "Logistic Regression ElasticNet"]}

    hmm_cols_fixed = None
    n_states_per_fold = []
    n_folds_total = N_SPLITS * N_REPEATS
    for fold_i, (tr_idx, te_idx) in enumerate(cv.split(sample_ids, y)):
        print(f"\n  === Fold {fold_i + 1}/{n_folds_total} ===", flush=True)
        train_sids = [sample_ids[i] for i in tr_idx]

        # Fold-specific HMM features
        print(f"    Training HMM on {len(train_sids)} train samples "
              f"(per-fold BIC={PER_FOLD_BIC})...", flush=True)
        feat, n_states_fold = fold_hmm_features(train_sids, sample_ids,
                                                  sample_chrom, labels_dict)
        n_states_per_fold.append(n_states_fold)
        X_hmm, hmm_cols = align_features(feat, sample_ids,
                                          feature_cols=hmm_cols_fixed)
        if hmm_cols_fixed is None:
            hmm_cols_fixed = hmm_cols
            print(f"    HMM feature columns: {len(hmm_cols_fixed)}", flush=True)

        # Classify
        X_hmm_tr, X_hmm_te = X_hmm[tr_idx], X_hmm[te_idx]
        X_raw_tr, X_raw_te = X_raw_full[tr_idx], X_raw_full[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        evaluate(X_hmm_tr, X_hmm_te, y_tr, y_te, hmm_res)
        evaluate(X_raw_tr, X_raw_te, y_tr, y_te, raw_res)

        if (fold_i + 1) % 5 == 0:
            # Progress snapshot of AUC so far
            for name in hmm_res:
                m_hmm = np.mean(hmm_res[name]["auc"])
                m_raw = np.mean(raw_res[name]["auc"])
                print(f"    [progress] {name}: HMM={m_hmm:.3f} Raw={m_raw:.3f}",
                      flush=True)

    # Per-fold n_states distribution (informative if PER_FOLD_BIC=True)
    if PER_FOLD_BIC:
        from collections import Counter
        dist = Counter(n_states_per_fold)
        print(f"\n  === Per-fold n_states (BIC selection) ===")
        for n in sorted(dist):
            print(f"    {n} states: {dist[n]}/{len(n_states_per_fold)} folds "
                  f"({100 * dist[n] / len(n_states_per_fold):.0f}%)")
        pd.DataFrame({"fold": range(1, len(n_states_per_fold) + 1),
                       "n_states": n_states_per_fold}).to_csv(
            os.path.join(RESULTS_DIR, "foldwise_bic_states.csv"), index=False)

    # Summarize
    all_rows = summarize(hmm_res, "HMM_foldwise")
    all_rows += summarize(raw_res, "Raw")
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(RESULTS_DIR, "classification_foldwise.csv"),
              index=False)

    # Paired test HMM vs Raw per model
    print("\n  === HMM_foldwise vs Raw (paired CV folds) ===")
    rows = []
    for model in hmm_res:
        for metric in ["auc", "pr_auc", "f1", "bal_acc"]:
            a = np.array(hmm_res[model][metric])
            b = np.array(raw_res[model][metric])
            diff = float(np.mean(a) - np.mean(b))
            t_stat, t_p = stats.ttest_rel(a, b)
            w_stat, w_p = stats.wilcoxon(a, b)
            winner = "HMM_foldwise" if diff > 0 else "Raw"
            sig = "***" if t_p < 0.001 else "**" if t_p < 0.01 else "*" if t_p < 0.05 else "ns"
            print(f"  {model:<24} {metric.upper():<8} diff={diff:+.4f}  p={t_p:.4f} {sig}  ({winner})")
            rows.append({"model": model, "metric": metric.upper(),
                          "diff": diff, "t_p": t_p, "w_p": w_p})
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, "statistical_tests_foldwise.csv"),
        index=False)

    print(f"\n  Saved: {RESULTS_DIR}/classification_foldwise.csv")
    print("  Done.")
