"""Genomic bin framework for cross-platform normalization.

Defines fixed 1Mb bins across hg19 autosomes (chr1-22) and provides
functions to map probes from any platform to a common coordinate space.
"""

import numpy as np
import pandas as pd

BIN_SIZE = 1_000_000  # 1 Mb

# hg19 chromosome lengths (autosomes only)
CHROM_LENGTHS = {
    "chr1": 249_250_621, "chr2": 243_199_373, "chr3": 198_022_430,
    "chr4": 191_154_276, "chr5": 180_915_260, "chr6": 171_115_067,
    "chr7": 159_138_663, "chr8": 146_364_022, "chr9": 141_213_431,
    "chr10": 135_534_747, "chr11": 135_006_516, "chr12": 133_851_895,
    "chr13": 115_169_878, "chr14": 107_349_540, "chr15": 102_531_392,
    "chr16": 90_354_753, "chr17": 81_195_210, "chr18": 78_077_248,
    "chr19": 59_128_983, "chr20": 63_025_520, "chr21": 48_129_895,
    "chr22": 51_304_566,
}

# hg19 centromere positions (approximate midpoints)
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

CHROMOSOMES = [f"chr{i}" for i in range(1, 23)]


def _build_bin_table():
    """Build a table of all genomic bins with their coordinates and arm assignments.

    Returns:
        DataFrame with columns: bin_idx, chrom, start, end, midpoint, arm, arm_id
    """
    rows = []
    idx = 0
    for chrom in CHROMOSOMES:
        length = CHROM_LENGTHS[chrom]
        centro = CENTROMERES[chrom]
        for start in range(0, length, BIN_SIZE):
            end = min(start + BIN_SIZE, length)
            midpoint = (start + end) // 2
            arm = "p" if midpoint < centro else "q"
            rows.append({
                "bin_idx": idx,
                "chrom": chrom,
                "start": start,
                "end": end,
                "midpoint": midpoint,
                "arm": arm,
                "arm_id": chrom + arm,
            })
            idx += 1
    return pd.DataFrame(rows)


# Module-level bin table (built once on import)
BIN_TABLE = _build_bin_table()
N_BINS = len(BIN_TABLE)

# Lookup structures for fast probe-to-bin mapping
_chrom_bin_starts = {}  # chrom -> array of bin start positions
_chrom_bin_offset = {}  # chrom -> offset into global bin index
_offset = 0
for chrom in CHROMOSOMES:
    mask = BIN_TABLE["chrom"] == chrom
    starts = BIN_TABLE.loc[mask, "start"].values
    _chrom_bin_starts[chrom] = starts
    _chrom_bin_offset[chrom] = _offset
    _offset += len(starts)


def assign_probe_to_bin(chrom, position):
    """Map a single probe to its bin index.

    Args:
        chrom: Chromosome name (e.g., "chr1")
        position: Genomic position in base pairs

    Returns:
        Global bin index, or -1 if chromosome not recognized
    """
    if chrom not in _chrom_bin_starts:
        return -1
    local_idx = int(position // BIN_SIZE)
    n_bins_chrom = len(_chrom_bin_starts[chrom])
    if local_idx >= n_bins_chrom:
        local_idx = n_bins_chrom - 1
    return _chrom_bin_offset[chrom] + local_idx


def compute_bin_values(probes_df):
    """Compute median log2 ratio per genomic bin for one sample.

    Args:
        probes_df: DataFrame with columns [chr, start, LogRatio]
                   (all probes from one sample)

    Returns:
        1D numpy array of shape (N_BINS,) with median log2 ratio per bin.
        Bins with no probes get NaN.
    """
    bin_values = np.full(N_BINS, np.nan)
    # Accumulate probes per bin
    bin_probes = [[] for _ in range(N_BINS)]

    for chrom in probes_df["chr"].unique():
        if chrom not in _chrom_bin_starts:
            continue
        cdf = probes_df[probes_df["chr"] == chrom]
        positions = cdf["start"].values
        log_ratios = cdf["LogRatio"].values
        offset = _chrom_bin_offset[chrom]
        n_bins_chrom = len(_chrom_bin_starts[chrom])

        local_indices = np.minimum(positions // BIN_SIZE, n_bins_chrom - 1).astype(int)
        for i in range(len(local_indices)):
            bin_probes[offset + local_indices[i]].append(log_ratios[i])

    for i in range(N_BINS):
        if bin_probes[i]:
            bin_values[i] = np.median(bin_probes[i])

    return bin_values


def bins_to_arm_mapping():
    """Return a dict mapping each bin index to its chromosome arm ID.

    Returns:
        dict of {bin_idx: arm_id} (e.g., {0: "chr1p", 125: "chr1q", ...})
    """
    return dict(zip(BIN_TABLE["bin_idx"], BIN_TABLE["arm_id"]))


def get_chrom_bins(chrom):
    """Get bin indices and midpoints for a specific chromosome.

    Args:
        chrom: Chromosome name (e.g., "chr1")

    Returns:
        Tuple of (bin_indices, midpoints) as numpy arrays
    """
    mask = BIN_TABLE["chrom"] == chrom
    return (BIN_TABLE.loc[mask, "bin_idx"].values,
            BIN_TABLE.loc[mask, "midpoint"].values)


def binned_to_chrom_arrays(bin_values):
    """Split a genome-wide bin array into per-chromosome arrays.

    Args:
        bin_values: 1D array of shape (N_BINS,)

    Returns:
        dict of {chrom: (log_ratios, midpoints, bin_indices)} for chromosomes
        with at least one non-NaN bin
    """
    result = {}
    for chrom in CHROMOSOMES:
        indices, midpoints = get_chrom_bins(chrom)
        values = bin_values[indices]
        valid = ~np.isnan(values)
        if valid.sum() > 0:
            result[chrom] = (values[valid], midpoints[valid], indices[valid])
    return result
