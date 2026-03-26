###############################################################################
# HMM core functions — shared between 02_hmm_segmentation.py and
# 01_process_all_datasets.py
#
# v2: Optimized parameters for MGUS detection
#   - Lower MIN_MEAN_SEP (0.05) to catch subtle MGUS CNAs
#   - Less sticky transitions (0.995) to improve state switching
#   - Pooled transition matrix support for stable cross-chromosome estimates
#   - Lower default MIN_CNA_PROBES (5) to preserve focal events
###############################################################################

import warnings
import numpy as np
from hmmlearn.hmm import GaussianHMM

N_STATES = 3
MIN_MEAN_SEP = 0.05  # was 0.10 — lowered to detect subtle MGUS CNAs
DEFAULT_MIN_CNA_PROBES = 5  # was 10 — lowered to preserve focal events

# Less sticky transitions: 0.995 self-transition (was 0.999)
# Allows the HMM to detect shorter CNA segments and arm-level events
# with gradual boundaries
DEFAULT_TRANSMAT = np.array([
    [0.995, 0.004, 0.001],
    [0.002, 0.995, 0.003],
    [0.001, 0.004, 0.995],
])

DEFAULT_STARTPROB = np.array([0.05, 0.90, 0.05])
DEFAULT_MEANS = np.array([[-0.3], [0.0], [0.3]])
DEFAULT_COVARS = np.array([[0.03], [0.005], [0.03]])


def fit_hmm_chromosome(args, transmat=None):
    """Two-pass 3-state Gaussian HMM for one chromosome.

    Pass 1: Learn all parameters from data.
    Pass 2: Validate mean separation — collapse states too close to neutral.

    If transmat is provided (pooled mode), it is used as the fixed transition
    matrix and not updated during fitting.
    """
    chrom, log_ratios, starts, ends = args

    if len(log_ratios) < 10:
        return None

    X = log_ratios.reshape(-1, 1)

    use_pooled = transmat is not None
    params_to_fit = "smc" if use_pooled else "stmc"

    model = GaussianHMM(
        n_components=N_STATES,
        covariance_type="diag",
        n_iter=200,
        tol=0.001,
        random_state=42,
        init_params="",
        params=params_to_fit,
    )
    model.startprob_ = DEFAULT_STARTPROB.copy()
    model.transmat_ = transmat.copy() if use_pooled else DEFAULT_TRANSMAT.copy()
    model.means_ = DEFAULT_MEANS.copy()
    model.covars_ = DEFAULT_COVARS.copy()

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X)
    except Exception:
        return None

    if np.any(np.isnan(model.means_)) or np.any(np.isnan(model.startprob_)):
        states = np.ones(len(log_ratios), dtype=int)
        return chrom, states, np.array([-0.3, 0.0, 0.3]), np.array([0.03, 0.005, 0.03])

    order = np.argsort(model.means_.flatten())
    sorted_means = model.means_.flatten()[order]
    sorted_covars = model.covars_.flatten()[order]

    del_sep = sorted_means[1] - sorted_means[0]
    amp_sep = sorted_means[2] - sorted_means[1]

    try:
        states = model.predict(X)
    except Exception:
        states = np.ones(len(log_ratios), dtype=int)
        return chrom, states, sorted_means, sorted_covars

    remap = np.zeros(N_STATES, dtype=int)
    for new_idx, old_idx in enumerate(order):
        remap[old_idx] = new_idx
    states = remap[states]

    if del_sep < MIN_MEAN_SEP:
        states[states == 0] = 1
    if amp_sep < MIN_MEAN_SEP:
        states[states == 2] = 1

    return chrom, states, sorted_means, sorted_covars


def estimate_pooled_transmat(all_log_ratios):
    """Estimate a shared transition matrix from all chromosomes in a sample.

    First pass: fit each chromosome independently, collect transition counts.
    Second pass: normalize to get a pooled transition matrix.

    Args:
        all_log_ratios: dict of chrom -> log_ratio array

    Returns:
        Pooled transition matrix (3x3 numpy array)
    """
    trans_counts = np.zeros((N_STATES, N_STATES))

    for chrom, lr in all_log_ratios.items():
        if len(lr) < 10:
            continue

        X = lr.reshape(-1, 1)
        model = GaussianHMM(
            n_components=N_STATES,
            covariance_type="diag",
            n_iter=100,
            tol=0.01,
            random_state=42,
            init_params="",
            params="stmc",
        )
        model.startprob_ = DEFAULT_STARTPROB.copy()
        model.transmat_ = DEFAULT_TRANSMAT.copy()
        model.means_ = DEFAULT_MEANS.copy()
        model.covars_ = DEFAULT_COVARS.copy()

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X)
            states = model.predict(X)
        except Exception:
            continue

        # Reorder states by mean
        order = np.argsort(model.means_.flatten())
        remap = np.zeros(N_STATES, dtype=int)
        for new_idx, old_idx in enumerate(order):
            remap[old_idx] = new_idx
        states = remap[states]

        # Accumulate transition counts
        for i in range(len(states) - 1):
            trans_counts[states[i], states[i + 1]] += 1

    # Normalize rows
    row_sums = trans_counts.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # avoid division by zero
    pooled = trans_counts / row_sums

    # Ensure valid (no zero rows)
    for i in range(N_STATES):
        if pooled[i].sum() < 0.01:
            pooled[i] = DEFAULT_TRANSMAT[i]

    return pooled


def postprocess_states(states, min_cna_probes=None):
    """Filter short CNA runs back to neutral."""
    if min_cna_probes is None:
        min_cna_probes = DEFAULT_MIN_CNA_PROBES
    states = states.copy()
    i = 0
    while i < len(states):
        if states[i] != 1:
            j = i
            while j < len(states) and states[j] == states[i]:
                j += 1
            if (j - i) < min_cna_probes:
                states[i:j] = 1
            i = j
        else:
            i += 1
    return states
