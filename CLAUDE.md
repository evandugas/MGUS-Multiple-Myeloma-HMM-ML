# MM/MGUS Classification Project

## Overview
MA770 final project (Evan Dugas & Fred Choi). The central question: **Does cleaning up genomic noise with an HMM produce better features for predicting cancer stage (MGUS vs MM) than using raw copy number data directly?**

Two-dataset analysis (GSE77975 + GSE33685, ~179 usable samples: 96 MGUS + 83 MM). Custom 3-state HMM segmentation on 1Mb-binned data produces arm-level CNA features, compared against raw log2 ratio features for classification. Both datasets are Agilent aCGH.

## Data

### Dataset Summary
| Dataset | Platform | MGUS | MM | Excluded | Notes |
|---------|----------|------|----|----------|-------|
| GSE77975 | Agilent GPL11358 (1M) | 90 | 16 | 30 | Primary dataset |
| GSE33685 | Agilent GPL10152 (60K) | 6 | 67 | 0 | Japanese cohort |
| **Total** | | **96** | **83** | **30** | |

### Setup (from scratch)
```bash
bash pipeline/00_download.sh           # Download + extract data
py -3 pipeline/01_sort_samples.py     # Sort into mgus/mm/excluded
py -3 pipeline/05_process.py          # Parse -> bin -> HMM -> arm features -> merge
py -3 pipeline/06_classify.py         # Classification + stat tests
py -3 pipeline/07_deep_learning.py    # MLP comparison
py -3 pipeline/08_analyze.py          # SHAP + cross-dataset + CNA heatmap
```

## Pipeline
```
Per-dataset (independent):
  Raw Agilent .gz files -> parse -> clean
  -> Map probes to 1Mb genomic bins (2,897 bins, hg19) -> median per bin
  -> Per-sample centering -> pooled 3-state HMM on binned data
  -> arm-level CNA fractions + derived features (burden, segments, means)

Merge:
  214 arm-level features + ~2,753 bin-level features per sample
  -> single master matrix

Classification:
  4 feature sets: HMM basic (84) vs HMM enhanced (214) vs Raw (88) vs Combined (302)
  Models: RF + LR + MLP with 5x10 repeated stratified CV
  + Leave-one-dataset-out CV for cross-dataset generalization
```

## HMM Details

### Parameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| MIN_MEAN_SEP | 0.05 | Detect subtle MGUS CNAs |
| MIN_CNA_PROBES | 5 | Preserve focal events |
| Self-transition | 0.995 | Better state switching |
| Transition matrix | Pooled per-sample | Stable cross-chromosome estimates |

### Feature Sets
- **HMM Basic** (84): del/amp fractions per arm
- **HMM Enhanced** (214): basic + CNA burden + segment counts + segment mean log-ratios
- **Raw** (88): mean + SD of raw LogRatio per arm
- **Combined** (302): HMM Enhanced + Raw stacked

## Known Issues

| Issue | Location | Severity | Effect |
|-------|----------|----------|--------|
| **Double centering** | 05_process.py: clean_probe_data + _bin_one_sample | High | Probe-level median centering THEN bin-level median centering attenuates real CNA amplitude |
| **1Mb bins too coarse** | 02_genomic_bins.py | High | Focal MGUS CNAs (<1Mb) lost in median aggregation |
| **MIN_CNA_PROBES=5 -> 5Mb min segment** | 04_hmm_core.py | High | At 1Mb bins, 5 bins = 5Mb minimum segment |
| **Hard-coded HMM means [-0.3, 0, 0.3]** | 04_hmm_core.py | Medium | Doesn't match actual per-platform signal ranges |
| **`vals != 0` bug in raw features** | 06_classify.py: build_raw_arm_features | Bug | Removes true zero-valued bins instead of NaN-only |
| **NaN bins dropped before HMM** | 05_process.py | Medium | Creates discontinuities in HMM time series |

## Project Plan

### Completed
- [x] Phase 1: Data prep (GSE77975 GPL11358)
- [x] Phase 2: 3-state HMM (two-pass, Baum-Welch + Viterbi)
- [x] Phase 3: Arm-level features + Fisher/Wilcoxon tests
- [x] Phase 4-5: Classification (HMM vs raw comparison)
- [x] Phase 6: Dataset expansion (GSE77975 + GSE33685)
- [x] Phase 7: HMM optimization (pooled training, lower thresholds, derived features)
- [x] Phase 8: Deep learning + SHAP + cross-dataset + statistical tests

### Phase 9: HMM Improvements (TODO)
1. Fix `vals != 0` bug in raw features
2. Remove double centering in `_bin_one_sample`
3. Adaptive HMM initialization (per-sample quantile-based means)
4. Finer bin resolution (500Kb / 250Kb)
5. Mean instead of median bin aggregation

### Phase 10: Report & presentation

## Scripts
All scripts live in `pipeline/`. Run numbered scripts in order (02-04 are library modules).

| File | Purpose |
|------|---------|
| `00_download.sh` | Download GSE77975 + GSE33685 |
| `01_sort_samples.py` | Sort samples by MGUS/MM/excluded label |
| `02_genomic_bins.py` | 1Mb bin framework, probe-to-bin mapping, arm assignments |
| `03_parsers.py` | Agilent aCGH parser |
| `04_hmm_core.py` | Shared HMM functions (fit, postprocess, pooled transmat) |
| `05_process.py` | Parse -> bin -> HMM -> arm features -> merge |
| `06_classify.py` | Classification (4 feature sets, LODO CV, stat tests) |
| `07_deep_learning.py` | MLP vs RF vs LR comparison |
| `08_analyze.py` | SHAP, cross-dataset, CNA heatmap, platform QC |

## Output Files
```
output/plots/
  01-05: Classification plots (AUC, ROC, importance, metrics, confusion)
  06: CNA landscape heatmap
  07: MLP vs RF vs LR comparison
  08-10: SHAP (beeswarm, bar, waterfall)
  11: Leave-one-dataset-out CV
  12: Platform QC

output/features/
  feature_matrix_arm_all.csv      # Arm-level features (master matrix)
  feature_matrix_binned.csv       # Bin-level log2 ratios
  feature_matrix_raw_arm.csv      # Raw arm mean+SD features

output/results/
  classification_results.csv      # Main results table
  statistical_tests.csv           # p-values for pairwise comparisons
  lodo_results.csv                # Leave-one-dataset-out results
  deep_learning_results.csv       # MLP comparison
  cross_dataset_results.csv       # Cross-dataset validation
```

## Key Biology Context
- MM: all have >=1 CNA, median 13 CNAs/patient
- MGUS: 65.6% have >=1 CNA, median 3 CNAs/patient
- Known differential regions: losses at 1p, 8p, 12p, 13q; gain at 1q
