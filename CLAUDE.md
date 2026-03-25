# MM/MGUS Classification Project

## Overview
MA770 final project (Evan Dugas & Fred Choi). The central question: **Does cleaning up genomic noise with an HMM produce better features for predicting cancer stage (MGUS vs MM) than using raw copy number data directly?**

Re-analysis of GSE77975 (90 MGUS + 33 MM patients, Agilent CGH+SNP arrays) — replacing Agilent's black-box CNA calling with a custom HMM, then comparing HMM-derived features vs raw log2 ratio features for classification.

## Data
- **Source**: GEO accession [GSE77975](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE77975)
- **Reference paper**: Lopez-Corral et al. (2016) — https://onlinelibrary.wiley.com/doi/10.1111/ejh.12774
- **Directory structure**:
  - `data/GSE77975_RAW.tar` — original tarball (136 gzipped TXT files)
  - `data/raw/mgus/` — 90 MGUS tumor samples
  - `data/raw/mm/` — 33 MM tumor samples
  - `data/raw/normal/` — 13 paired normal samples (exclude from classification)
  - `data/metadata/` — GEO series matrix files (all 4 platforms)
    - `GSE77975-GPL11358_series_matrix.txt.gz` — 119 samples (90 MGUS + 13 normals labeled MGUS + 16 MM)
    - `GSE77975-GPL16237_series_matrix.txt.gz` — 7 MM samples
    - `GSE77975-GPL10152_series_matrix.txt.gz` — 5 MM samples
    - `GSE77975-GPL21461_series_matrix.txt.gz` — 5 MM samples
- **File format**: Agilent Feature Extraction output (tab-delimited, gzipped)
  - Header sections: FEPARAMS (scan metadata), STATS (QC metrics), then FEATURES (probe data)
  - ~159,777 non-control probes per sample (ControlType == 0)
  - Key columns in FEATURES section: `ProbeName`, `SystematicName` (e.g. `chr12:24273514-24273573`), `LogRatio`, `LogRatioError`, `PValueLogRatio`, `gProcessedSignal`, `rProcessedSignal`
  - Control probes (ControlType != 0): `HsCGHBrightCorner`, `DarkCorner2`, etc. — filter these out
- **Labels**: All verified from official GEO series matrix metadata across 4 platforms
  - 13 normals in GPL11358 are mislabeled as MGUS — identified and separated by paired-normal metadata

## Project Plan

### Phase 1: Data Preparation
- [ ] Load all 123 sample files and attach MGUS/MM labels from GEO metadata
- [ ] EDA: data quality, batch effects, signal distributions
- [ ] Cleaning: filter bad probes, median-center, drop sex chromosomes, remove common CNV regions

### Phase 2: HMM Implementation & Segmentation
- [ ] Implement HMM with 3 hidden states: deletion (<2 copies), neutral (2 copies), amplification (>2 copies) — possibly extend to 4 states with high-level amplification
- [ ] Gaussian emission distributions conditioned on hidden state
- [ ] Estimate parameters with Baum-Welch algorithm
- [ ] Decode most probable state sequences with Viterbi algorithm
- [ ] Run HMM per chromosome per sample
- [ ] Visualize and validate segmentation on selected chromosomes

### Phase 3: Feature Construction & Statistical Analysis
- [ ] Build sample-by-region feature matrix (rows = patients, columns = genomic windows, values = frequency of deletion/amplification state)
- [ ] Fisher's exact test with Benjamini-Hochberg FDR correction to find regions associated with MGUS-to-MM progression
- [ ] Sanity check: recovered regions should overlap with paper's Table 1 (losses at 1p, 8p, 12p, 13q; gain at 1q)

### Phase 4: Machine Learning Classification
- [ ] Primary model: Random Forest on HMM-derived features
- [ ] Baseline: Logistic Regression with L1 regularization
- [ ] Optional: Gradient-boosted classifier if sample size permits
- [ ] k-fold cross-validation; metrics: AUC-ROC, F1-score, balanced accuracy
- [ ] Aggregate feature importance across folds to identify most predictive regions

### Phase 5: Comparative Analysis
- [ ] Repeat identical classification pipeline using raw log2 ratios as features (no HMM)
- [ ] Compare HMM-based vs raw-signal classification within same CV framework
- [ ] Assess whether probabilistic segmentation improves classification

### Phase 6: Report & Presentation
- [ ] Interpret results in context of known myeloma biology
- [ ] Figures and final writeup

## Tech Stack
- **R** — Data loading, EDA, statistical testing
  - GEOquery (fetch sample metadata/labels)
  - ggplot2 (genomic visualizations)
  - Fisher's exact test, BH correction
- **Python** — HMM, feature construction, ML classification
  - hmmlearn (HMM: Baum-Welch, Viterbi)
  - scikit-learn (Random Forest, Logistic Regression L1, cross-validation, AUC-ROC)
  - pandas/numpy (data wrangling, feature matrices)
- **Handoff**: R saves cleaned data as CSV/RDS, Python reads it for modeling

## Key Biology Context
- MM patients: all have >= 1 CNA, median 13 CNAs/patient
- MGUS patients: 65.6% have >= 1 CNA, median 3 CNAs/patient
- Known differential regions: losses at 1p, 8p, 12p, 13q; gain at 1q (use as validation)
