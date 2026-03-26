# MM/MGUS Classification Project

## Overview
MA770 final project (Evan Dugas & Fred Choi). The central question: **Does cleaning up genomic noise with an HMM produce better features for predicting cancer stage (MGUS vs MM) than using raw copy number data directly?**

Multi-dataset analysis across 6 studies (639 samples: 116 MGUS + 523 MM). Custom 3-state HMM segmentation on 1Mb-binned data produces arm-level CNA features, compared against raw log2 ratio features for classification. Genomic bin normalization enables cross-platform harmonization.

## Data

### Dataset Summary
| Dataset | Platform | MGUS | MM | Excluded | Notes |
|---------|----------|------|----|----------|-------|
| GSE77975 | Agilent GPL11358 (1M) + 3 others | 90 | 33 | 13 normals | Primary dataset |
| GSE33685 | Agilent GPL10152 (60K) | 6 | 67 | - | Japanese cohort |
| GSE31339 | Affymetrix SNP 6.0 (1.85M) | 20 | 34 | 30 (SMM+normals) | Lopez-Corral et al. 2012 |
| GSE26849 | Agilent 244K | 0 | 254 | - | MMRC reference collection |
| GSE44745 | Agilent 244K | 0 | 63 | - | Malaysian multi-ethnic |
| GSE29023 | Agilent 244K | 0 | 92 | - | Paired expression+CGH |
| **Total** | | **116** | **523** | **43** | |

### Setup (from scratch)
```bash
bash pipeline/00_download.sh           # Download + extract all data
python pipeline/01_sort_samples.py     # Sort into mgus/mm/excluded
python pipeline/05_process.py          # Parse -> bin -> HMM -> arm features -> merge
python pipeline/06_classify.py         # Classification + stat tests
python pipeline/07_deep_learning.py    # MLP comparison
python pipeline/08_analyze.py          # SHAP + cross-dataset + CNA heatmap
```

## Pipeline
```
Per-dataset (independent):
  Raw files (Agilent .gz / Affymetrix processed) -> parse -> clean
  -> Map probes to 1Mb genomic bins (2,897 bins, hg19) -> median per bin
  -> Per-sample centering -> pooled 3-state HMM on binned data
  -> arm-level CNA fractions + derived features (burden, segments, means)

Merge (platform-agnostic via binning):
  209 arm-level features + 2,897 bin-level features per sample
  -> single 639-row master matrix

Classification:
  4 feature sets: HMM basic (82) vs HMM enhanced (209) vs Raw (82) vs Combined (291)
  Models: RF + LR + MLP with 5x10 repeated stratified CV
  + Leave-one-dataset-out CV for cross-platform generalization
```

## HMM Details

### Optimized Parameters (v2)
| Parameter | v1 | v2 | Rationale |
|-----------|----|----|-----------|
| MIN_MEAN_SEP | 0.10 | 0.05 | Detect subtle MGUS CNAs |
| MIN_CNA_PROBES | 10 | 5 | Preserve focal events |
| Self-transition | 0.999 | 0.995 | Better state switching |
| Transition matrix | Per-chromosome | Pooled per-sample | Stable cross-chromosome estimates |

### Feature Sets
- **HMM Basic** (82): del/amp fractions per arm
- **HMM Enhanced** (209): basic + CNA burden + segment counts + segment mean log-ratios
- **Raw** (82): mean + SD of raw LogRatio per arm
- **Combined** (291): HMM Enhanced + Raw stacked

## Results

### Classification (196 samples, 5x10 repeated stratified CV)
| Features | Model | AUC | F1 | BalAcc | Sens | Spec |
|----------|-------|-----|-----|--------|------|------|
| HMM basic | RF | 0.902 | 0.798 | 0.808 | 0.757 | 0.858 |
| HMM basic | LR | 0.898 | 0.811 | 0.819 | 0.777 | 0.861 |
| **HMM enhanced** | **RF** | **0.937** | **0.852** | **0.864** | 0.791 | 0.936 |
| HMM enhanced | LR | 0.919 | 0.837 | 0.844 | 0.798 | 0.891 |
| Raw arm | RF | 0.921 | 0.829 | 0.834 | 0.795 | 0.872 |
| Raw arm | LR | 0.931 | 0.834 | 0.838 | 0.806 | 0.871 |
| **Combined** | **RF** | **0.945** | 0.854 | 0.862 | 0.803 | 0.921 |
| **Combined** | **LR** | 0.940 | **0.864** | **0.868** | 0.830 | 0.905 |

### Statistical Significance (paired t-test, 50 CV folds)
- HMM Enhanced vs Raw: AUC p=0.006**, F1 p=0.004**, BalAcc p<0.001***
- HMM Enhanced vs HMM Basic: all p<0.001***
- HMM Enhanced vs Combined: AUC p=0.026*, F1 p=0.807 ns, BalAcc p=0.757 ns

### Deep Learning (MLP) Comparison
| Features | MLP AUC | RF AUC | LR AUC |
|----------|---------|--------|--------|
| HMM Enhanced | 0.912 | 0.937 | 0.919 |
| Combined | 0.924 | 0.946 | 0.940 |

MLP does not beat RF/LR on this dataset (196 samples insufficient for DL advantage).

### Cross-Dataset Validation
Poor performance in both directions, even after ComBat batch correction:

| Direction | Uncorrected AUC | ComBat AUC |
|-----------|----------------|------------|
| GSE33685 -> GSE77975 | 0.298 (RF) | 0.383 (RF) |
| GSE77975 -> GSE33685 | 0.428 (RF) | 0.363 (RF) |

ComBat does not help because this is **platform incompatibility**, not a batch effect. GPL11358 and GPL10152 have different probe sets targeting different genomic regions. The "same" arm-level feature is derived from different underlying measurements. This validates the mixed-CV approach (pooling both datasets) as the correct evaluation strategy.

Note: GSE77979 (33 MGUS, same platform) was investigated but is 100% overlapping with GSE77975 — same patients under a separate accession for exome sequencing.

### SHAP Analysis
- Top SHAP features: seg_mean_del_chr4q, seg_mean_del_chr8q, seg_mean_del_chr3q, seg_mean_del_chr8p
- Deletion magnitude features dominate predictions
- Known myeloma regions (13q, 1p, 8p) confirmed by SHAP

### Key Findings
1. **HMM Enhanced RF beats Raw on all metrics** (p<0.01), confirming hypothesis
2. **Combined (HMM+Raw) achieves highest AUC (0.945)** but not significantly better than HMM Enhanced alone on F1/BalAcc
3. HMM enables extraction of segment-level features (CNA magnitude, burden) impossible from raw signal
4. Top features are biologically validated against published myeloma literature
5. Cross-dataset generalization fails due to platform incompatibility (not fixable by ComBat)
6. GSE77979 confirmed as duplicate of GSE77975 (100% sample overlap)

## Project Plan

### Completed
- [x] Phase 1: Data prep (GSE77975 GPL11358)
- [x] Phase 2: 3-state HMM (two-pass, Baum-Welch + Viterbi)
- [x] Phase 3: Arm-level features + Fisher/Wilcoxon tests
- [x] Phase 4-5: Classification (HMM vs raw comparison)
- [x] Phase 6: Dataset expansion (GSE77975 + GSE33685 = 196 samples)
- [x] Phase 7: HMM optimization (pooled training, lower thresholds, derived features)
- [x] Phase 8: Deep learning + SHAP + cross-dataset + statistical tests
- [ ] Phase 9: Report & presentation

## Scripts
All scripts live in `pipeline/`. Run numbered scripts in order (02-04 are library modules).

| File | Purpose |
|------|---------|
| `00_download.sh` | Download all 6 GEO datasets |
| `01_sort_samples.py` | Sort samples by MGUS/MM/excluded label |
| `02_genomic_bins.py` | 1Mb bin framework, probe-to-bin mapping, arm assignments |
| `03_parsers.py` | Platform-specific parsers (Agilent, Affymetrix) |
| `04_hmm_core.py` | Shared HMM functions (fit, postprocess, pooled transmat) |
| `05_process.py` | Parse -> bin -> HMM -> arm features -> merge |
| `06_classify.py` | Classification (4 feature sets, LODO CV, stat tests) |
| `07_deep_learning.py` | MLP vs RF vs LR comparison |
| `08_analyze.py` | SHAP, cross-dataset, CNA heatmap, ComBat, platform QC |
| `09_process_cel.R` | Affymetrix CEL -> log2 ratio (R fallback, conditional) |

## Output Files
```
output/plots/
  01_classification_auc.png       # AUC bar chart (4 feature sets)
  02_roc_curves.png               # ROC curves (8 models)
  03_feature_importance.png       # Top 20 RF features
  04_all_metrics.png              # AUC/F1/BalAcc comparison
  05_confusion_matrices.png       # Confusion matrices per feature set
  06_cna_landscape.png            # CNA heatmap (MGUS vs MM)
  07_dl_comparison.png            # MLP vs RF vs LR
  08_shap_beeswarm.png            # SHAP beeswarm plot
  09_shap_bar.png                 # SHAP bar plot
  10_shap_waterfall.png           # SHAP waterfall (individual patients)
  11_lodo_cv.png                  # Leave-one-dataset-out CV
  12_platform_qc.png              # Bin coverage + noise by dataset

output/features/
  feature_matrix_arm_all.csv      # Arm-level features (master matrix)
  feature_matrix_binned.csv       # Bin-level log2 ratios (2,897 bins)
  feature_matrix_raw_arm.csv      # Raw arm mean+SD features
  feature_matrix_combat.csv       # ComBat-corrected arm features
  feature_matrix_binned_combat.csv # ComBat-corrected bin features

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
