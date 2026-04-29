# HMM-Based Copy Number Classification: MGUS vs Multiple Myeloma

**MA770 Final Project** — Evan Dugas & Fred Choi

## Question

Does cleaning up genomic noise with a Hidden Markov Model produce better features for predicting cancer stage (MGUS vs MM) than using the raw copy number signal directly?

## Approach

We built a **cohort-trained Gaussian HMM** to segment probe-level aCGH copy number data into discrete copy-number states, then compared arm-level features derived from HMM segmentation against raw log2 ratio summary statistics for MGUS/MM classification.


## Setup

```bash
pip install numpy pandas scikit-learn scipy statsmodels hmmlearn matplotlib combat-pycombat

bash pipeline/00_download.sh
python pipeline/01_sort_samples.py
python pipeline/05_process.py              # DLRS QC + HMM + features
python pipeline/06_classify.py             # full-cohort classification + QC
python pipeline/07_classify_foldwise.py    # leakage-free HMM retraining + per-fold BIC
python pipeline/08_combat_analysis.py      # ComBat sensitivity analysis
```

## Pipeline

```
00_download.sh              Download GSE77975 from GEO
01_sort_samples.py          Sort into MGUS/MM/excluded; record platform per sample
02_genomic_bins.py          (library) Genomic bin utilities
03_parsers.py               (library) Agilent aCGH parser
04_hmm_core.py              BIC selection + cohort HMM (train + decode)
05_process.py               Parse -> DLRS QC -> HMM -> feature extraction
                            Saves cleaned probes (output/probes/) for 07
06_classify.py              Full-cohort CV (5 feature sets, 3 models)
                            + bootstrap CIs + permutation test
                            + operating-point analysis
                            + platform confound sensitivity
                            + PCA diagnostics + plots
07_classify_foldwise.py     Leakage-free: retrain HMM in each fold
                            + per-fold BIC state selection
08_combat_analysis.py       ComBat batch-correction sensitivity
```

## Output

```
output/features/    feature_matrix_hmm.csv, feature_matrix_raw.csv
output/qc/          sample_qc.csv (DLRS), dlrs_histogram.png
output/probes/      probes_clean.pkl (for fold-wise retrain)
output/results/
  classification_results.csv         full-cohort CV (5 feat x 3 models)
  classification_foldwise.csv        leakage-free CV (per-fold HMM + BIC)
  bootstrap_cis.csv                  95% CIs on AUC/PR-AUC/F1/BalAcc
  permutation_test.csv               null AUC + empirical p
  platform_confound.csv              GPL11358-only sensitivity
  combat_analysis.csv                ComBat-adjusted classification
  operating_points.csv               thresholds at 0.5 / Youden / sens>=0.80
  region_tests.csv                   per-arm Wilcoxon + BH
  statistical_tests.csv              HMM/HMM80 vs Raw paired (full cohort)
  statistical_tests_foldwise.csv     HMM vs Raw paired (fold-wise)
  foldwise_bic_states.csv            n_states selected per fold
  08_hmm_pca_coordinates.csv         HMM PCA coords
  09_raw_pca_coordinates.csv         Raw PCA coords
  10_hmm_cov_pca_coordinates.csv     HMM+Platform PCA coords
  11_raw_cov_pca_coordinates.csv     Raw+Platform PCA coords
output/plots/       01_auc, 02_roc, 03_importance, 04_metrics, 05_confusion,
                    06_cna_landscape, 07_arm_comparison, 08_permutation,
                    09_operating_points, 08/09/10/11 PCA diagnostics
                    (per-feature-set, by label and by platform)
```

## References

1. Aktas Samur A, et al. (2019). Deciphering the chronology of copy number alterations in multiple myeloma. Blood Cancer Journal, 9(4), 39. doi:10.1038/s41408-019-0199-3
2. Kyle RA, et al. (2010). Monoclonal gammopathy of undetermined significance (MGUS) and smoldering (asymptomatic) multiple myeloma: IMWG consensus perspectives. Leukemia, 24(6), 1121–1127. doi:10.1038/leu.2010.60
3. Hassan H, Szalat R. (2021). Genetic predictors of mortality in patients with multiple myeloma. The Application of Clinical Genetics, 14, 241–254. doi:10.2147/TACG.S262866
4. Stong N, et al. (2023). The location of the t(4;14) translocation breakpoint within the NSD2 gene identifies a subset of patients with high-risk NDMM. Blood, 141(13), 1574–1583. doi:10.1182/blood.2022016212
5. Mikulasova A, et al. (2016). Genome-wide profiling of copy-number alteration in MGUS. European Journal of Haematology. doi:10.1111/ejh.12774
6. Colella S, et al. (2007). QuantiSNP: an objective Bayes Hidden-Markov Model to detect and accurately map copy number variation. Nucleic Acids Research. doi:10.1093/nar/gkm076
7. Johnson WE, et al. (2007). Adjusting batch effects in microarray expression data using empirical Bayes methods (ComBat). Biostatistics. doi:10.1093/biostatistics/kxj037


