"""Gaussian HMM for copy number segmentation.

Two-stage cohort-trained approach:
  Stage 0: BIC model selection (3, 4, 5 states)
  Stage 1: Train shared emission parameters from all samples (Baum-Welch)
  Stage 2: Per-sample decoding with fixed emissions (Viterbi)
"""

import warnings
import numpy as np
from hmmlearn.hmm import GaussianHMM

# Minimum consecutive bins required to keep a CNA segment.
# Since we will decode 1 Mb bins, 3 bins ~= 3 Mb minimum event size.
MIN_CNA_BINS = 3

# Defaults for 3-state (overridden by BIC selection)
INIT_CONFIGS = {
    3: {
        "startprob": np.array([0.10, 0.80, 0.10]),
        "transmat": np.array([
            [0.995, 0.004, 0.001],
            [0.002, 0.995, 0.003],
            [0.001, 0.004, 0.995]]),
        "means": np.array([[-0.3], [0.0], [0.3]]),
        "covars": np.array([[0.03], [0.005], [0.03]]),
    },
    4: {
        "startprob": np.array([0.05, 0.10, 0.75, 0.10]),
        "transmat": np.array([
            [0.990, 0.008, 0.001, 0.001],
            [0.004, 0.990, 0.005, 0.001],
            [0.001, 0.004, 0.990, 0.005],
            [0.001, 0.001, 0.008, 0.990]]),
        "means": np.array([[-0.5], [-0.15], [0.0], [0.2]]),
        "covars": np.array([[0.05], [0.01], [0.005], [0.02]]),
    },
    5: {
        "startprob": np.array([0.02, 0.08, 0.76, 0.10, 0.04]),
        "transmat": np.array([
            [0.990, 0.008, 0.001, 0.001, 0.000],
            [0.004, 0.990, 0.005, 0.001, 0.000],
            [0.001, 0.003, 0.990, 0.005, 0.001],
            [0.000, 0.001, 0.005, 0.990, 0.004],
            [0.000, 0.001, 0.001, 0.008, 0.990]]),
        "means": np.array([[-0.5], [-0.15], [0.0], [0.15], [0.5]]),
        "covars": np.array([[0.05], [0.01], [0.005], [0.01], [0.05]]),
    },
}


def _fit_model(n_states, X, lengths):
    """Fit a single HMM with n_states and return (model, log_likelihood)."""
    cfg = INIT_CONFIGS[n_states]
    model = GaussianHMM(
        n_components=n_states, covariance_type="diag",
        n_iter=100, tol=0.01, random_state=42,
        init_params="", params="stmc",
    )
    model.startprob_ = cfg["startprob"].copy()
    model.transmat_ = cfg["transmat"].copy()
    model.means_ = cfg["means"].copy()
    model.covars_ = cfg["covars"].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, lengths=lengths)

    ll = model.score(X, lengths=lengths)
    return model, ll


def select_n_states(all_sequences, max_sequences=300):
    """Stage 0: BIC model selection across 3, 4, 5 states.

    BIC = -2 * log_likelihood + n_params * ln(n_samples)
    Lower BIC = better model (balances fit vs complexity).
    """
    if len(all_sequences) > max_sequences:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_sequences), max_sequences, replace=False)
        seqs = [all_sequences[i] for i in idx]
    else:
        seqs = list(all_sequences)

    seqs = [s for s in seqs if len(s) >= 20]
    X = np.concatenate(seqs).reshape(-1, 1)
    lengths = [len(s) for s in seqs]
    n_obs = len(X)

    print(f"    BIC selection: {len(seqs)} sequences, {n_obs:,} probes", flush=True)

    best_n, best_bic = 3, np.inf
    for n in [3, 4, 5]:
        try:
            model, ll = _fit_model(n, X, lengths)
            # n_params = transitions + means + covars + startprob
            n_params = n * (n - 1) + n + n + (n - 1)
            bic = -2 * ll + n_params * np.log(n_obs)
            print(f"      {n} states: LL={ll:,.0f}  BIC={bic:,.0f}  "
                  f"params={n_params}", flush=True)
            if bic < best_bic:
                best_bic = bic
                best_n = n
        except Exception as e:
            print(f"      {n} states: failed ({e})", flush=True)

    print(f"    Selected: {best_n} states (lowest BIC)", flush=True)
    return best_n


def train_cohort_model(all_sequences, n_states, max_sequences=500):
    """Stage 1: Learn emission parameters from the full cohort.

    Returns (n_states, means, covars, transmat) with states ordered by mean.
    """
    if len(all_sequences) > max_sequences:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(all_sequences), max_sequences, replace=False)
        sequences = [all_sequences[i] for i in idx]
    else:
        sequences = list(all_sequences)

    sequences = [s for s in sequences if len(s) >= 20]
    cfg = INIT_CONFIGS[n_states]

    if not sequences:
        return n_states, cfg["means"].copy(), cfg["covars"].copy(), cfg["transmat"].copy()

    X = np.concatenate(sequences).reshape(-1, 1)
    lengths = [len(s) for s in sequences]

    print(f"    Training {n_states}-state HMM: {len(sequences)} sequences, "
          f"{len(X):,} probes", flush=True)

    try:
        model, _ = _fit_model(n_states, X, lengths)
    except Exception as e:
        print(f"    Training failed: {e}", flush=True)
        return n_states, cfg["means"].copy(), cfg["covars"].copy(), cfg["transmat"].copy()

    order = np.argsort(model.means_.flatten())
    means = model.means_[order].reshape(n_states, 1)
    covars = model.covars_[order].reshape(n_states, 1)
    transmat = model.transmat_[order][:, order]

    m = means.flatten()
    s = np.sqrt(covars.flatten())
    labels = ["del", "neut", "amp"] if n_states == 3 else \
             [f"s{i}" for i in range(n_states)]
    if n_states == 4:
        labels = ["deep_del", "del", "neut", "amp"]
    elif n_states == 5:
        labels = ["deep_del", "del", "neut", "gain", "high_amp"]

    print(f"    Learned states:", flush=True)
    for i in range(n_states):
        print(f"      {labels[i]:<10}: mean={m[i]:+.4f}  SD={s[i]:.4f}", flush=True)

    return n_states, means, covars, transmat


def decode_sample(chrom_data, n_states, cohort_means, cohort_covars,
                  cohort_transmat, sample_id):
    """Stage 2: Decode one sample with fixed cohort emissions.

    Fits per-sample transition matrix, then Viterbi decodes each chromosome.
    Returns list of (sample_id, chrom, start, log_ratio, state) tuples.
    State mapping: neutral = n_states // 2, del < neutral, amp > neutral.
    """
    valid = {c: d for c, d in chrom_data.items() if len(d[0]) >= 20}
    if not valid:
        return []

    cfg = INIT_CONFIGS[n_states]
    means = cohort_means.reshape(n_states, 1)
    covars = cohort_covars.reshape(n_states, 1)
    neutral = n_states // 2

    # Fit per-sample transitions
    trans_counts = np.zeros((n_states, n_states))
    for chrom, (lr, starts) in valid.items():
        model = GaussianHMM(
            n_components=n_states, covariance_type="diag",
            n_iter=50, tol=0.01, random_state=42,
            init_params="", params="st",
        )
        model.startprob_ = cfg["startprob"].copy()
        model.transmat_ = cohort_transmat.copy()
        model.means_ = means.copy()
        model.covars_ = covars.copy()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(lr.reshape(-1, 1))
            states = model.predict(lr.reshape(-1, 1))
            for i in range(len(states) - 1):
                trans_counts[states[i], states[i + 1]] += 1
        except Exception:
            continue

    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    sample_transmat = trans_counts / row_sums
    for i in range(n_states):
        if sample_transmat[i].sum() < 0.01:
            sample_transmat[i] = cohort_transmat[i]

    # Viterbi decode
    results = []
    for chrom, (lr, starts) in valid.items():
        model = GaussianHMM(
            n_components=n_states, covariance_type="diag",
            init_params="", params="",
        )
        model.startprob_ = cfg["startprob"].copy()
        model.transmat_ = sample_transmat.copy()
        model.means_ = means.copy()
        model.covars_ = covars.copy()

        try:
            states = model.predict(lr.reshape(-1, 1))
        except Exception:
            continue

        states = postprocess_states(states, neutral)
        # Remap to 3 categories: 0=del, 1=neutral, 2=amp
        for i in range(len(states)):
            if states[i] < neutral:
                cat = 0  # deletion
            elif states[i] > neutral:
                cat = 2  # amplification
            else:
                cat = 1  # neutral
            results.append((sample_id, chrom, int(starts[i]),
                            float(lr[i]), cat))

    return results


def postprocess_states(states, neutral_state, min_bins=MIN_CNA_BINS):
    """Filter short CNA segments back to neutral. States are assumed to come from fixed genomic bins, not raw probes. """
    states = states.copy()
    i = 0
    while i < len(states):
        if states[i] != neutral_state:
            j = i
            while j < len(states) and states[j] == states[i]:
                j += 1
            if (j - i) < min_bins:
                states[i:j] = neutral_state
            i = j
        else:
            i += 1
    return states
