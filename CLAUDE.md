# MM/MGUS Classification Project

## Overview
MA770 final project (Evan Dugas & Fred Choi). **Does HMM-based CNV segmentation produce better features for predicting cancer stage (MGUS vs MM) than using raw copy number data?**

GSE77975: 123 samples (90 MGUS + 33 MM) across 4 Agilent aCGH platforms. BIC-selected cohort-trained HMM on probe-level data produces arm-level CNA features, compared against raw log2 ratio features.

## Data
GSE77975 (Mikulasova et al. 2016): 4 Agilent platforms from one lab.

| Platform | MGUS | MM | Excluded |
|----------|------|----|----------|
| GPL11358 (CGH+SNP 1M) | 90 | 16 | 13 |
| GPL16237 (180K) | 0 | 7 | 0 |
| GPL10152 (60K) | 0 | 5 | 0 |
| GPL21461 (60K v2) | 0 | 5 | 0 |

## Setup
```bash
bash pipeline/00_download.sh       # Download GSE77975
py -3 pipeline/01_sort_samples.py  # Sort into mgus/mm/excluded
py -3 pipeline/05_process.py       # HMM segmentation + features
py -3 pipeline/06_classify.py      # Classification + tests + plots
```

## Pipeline
```
Parse Agilent .gz → clean → median-center

Stage 0: BIC Model Selection
  Compare 3, 4, 5-state HMMs on 300 sequences
  Select optimal state count (5 states selected by BIC)

Stage 1: Cohort HMM Training
  Fit N-state Gaussian HMM on ~2M probes from all samples (Baum-Welch)
  Learn shared emission means/covariances

Stage 2: Per-Sample Decoding
  Fix emissions, fit per-sample transitions, Viterbi decode each chromosome
  Remap to 3 categories (del/neutral/amp) → arm-level features

Raw Features: mean + SD of log2R per arm (from same probe data)

Classification: RF + LR L1 with 5×10 repeated stratified CV
Region Tests: Wilcoxon rank-sum on raw log2R per arm with BH correction
Model Comparison: Paired t-test + Wilcoxon on CV fold metrics
```

## HMM
| Parameter | Value |
|-----------|-------|
| States | BIC-selected (5: deep_del, del, neutral, gain, high_amp) |
| Input | Probe-level (~45K–108K probes/sample) |
| Training | Cohort-wide Baum-Welch on 500 sequences |
| Decoding | Per-sample transitions + Viterbi |
| MIN_CNA_PROBES | 5 |
| Output mapping | 5 states → 3 categories (del/neutral/amp) for features |

## Feature Sets
- **HMM** (~204): del/amp fractions + CNA burden + segment counts + segment means per arm
- **Raw** (80): mean + SD of raw log2R per arm

## Scripts
| File | Purpose |
|------|---------|
| `00_download.sh` | Download GSE77975 |
| `01_sort_samples.py` | Sort by label (all 4 platforms) |
| `02_genomic_bins.py` | Genomic bin utilities (library) |
| `03_parsers.py` | Agilent aCGH parser |
| `04_hmm_core.py` | BIC selection + cohort HMM (train + decode) |
| `05_process.py` | Parse → HMM → features |
| `06_classify.py` | Classification + tests + plots |

## Output
```
output/features/  feature_matrix_hmm.csv, feature_matrix_raw.csv
output/results/   classification_results.csv, region_tests.csv, statistical_tests.csv
output/plots/     01-07 (AUC, ROC, importance, metrics, confusion, CNA landscape, arm comparison)
```
