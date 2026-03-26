#!/bin/bash
###############################################################################
# Download and extract all GEO datasets for MM/MGUS classification project
#
# Datasets (Agilent CGH arrays — compatible with our custom HMM):
#   GSE77975 — 90 MGUS + 33 MM + 13 normals, Agilent GPL11358 + 3 other platforms
#   GSE33685 — 67 MM + 6 MGUS, Agilent GPL10152
#
# Total: 96 MGUS + 100 MM = 196 samples
#
# Usage: bash scripts/download_geo_data.sh
###############################################################################

set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$BASE_DIR/data"

download_dataset() {
    local acc="$1"
    local dest="$2"
    local desc="$3"
    local matrix_url="$4"

    echo ""
    echo "--- $acc ($desc) ---"
    mkdir -p "$dest" "$dest/raw"

    # Download RAW tar
    if [ ! -f "$dest/${acc}_RAW.tar" ]; then
        echo "  Downloading ${acc}_RAW.tar..."
        curl -L -o "$dest/${acc}_RAW.tar" \
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=${acc}&format=file"
    else
        echo "  ${acc}_RAW.tar already exists"
    fi

    # Extract tar into raw/
    local n_files
    n_files=$(find "$dest/raw" -maxdepth 1 -type f 2>/dev/null | wc -l)
    if [ "$n_files" -lt 2 ]; then
        echo "  Extracting..."
        tar -xf "$dest/${acc}_RAW.tar" -C "$dest/raw/"
        n_files=$(find "$dest/raw" -maxdepth 1 -type f 2>/dev/null | wc -l)
        echo "  Extracted $n_files files"
    else
        echo "  Already extracted ($n_files files)"
    fi

    # Download series matrix
    if [ ! -f "$dest/${acc}_series_matrix.txt.gz" ]; then
        echo "  Downloading series matrix..."
        curl -L -o "$dest/${acc}_series_matrix.txt.gz" "$matrix_url"
    else
        echo "  Series matrix already exists"
    fi
}

echo "================================================"
echo " MM/MGUS Project — GEO Data Download & Setup"
echo "================================================"
echo "Data directory: $DATA_DIR"

# === GSE77975: Primary dataset ===
echo ""
echo "====== GSE77975 (PRIMARY) ======"
download_dataset "GSE77975" "$DATA_DIR/GSE77975" \
    "90 MGUS + 33 MM + 13 normal, Agilent GPL11358" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77975/matrix/GSE77975-GPL11358_series_matrix.txt.gz"

# Download all 4 platform series matrices
for GPL in GPL11358 GPL16237 GPL10152 GPL21461; do
    if [ ! -f "$DATA_DIR/GSE77975/GSE77975-${GPL}_series_matrix.txt.gz" ]; then
        echo "  Downloading ${GPL} series matrix..."
        curl -L -o "$DATA_DIR/GSE77975/GSE77975-${GPL}_series_matrix.txt.gz" \
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77975/matrix/GSE77975-${GPL}_series_matrix.txt.gz"
    fi
done

# Sort GSE77975 into mgus/mm/mm_excluded/normal subdirectories
# (The R data prep script handles this based on series matrix labels)
for subdir in mgus mm mm_excluded normal; do
    mkdir -p "$DATA_DIR/GSE77975/raw/$subdir"
done
n_sorted=$(find "$DATA_DIR/GSE77975/raw/mgus" -name "GSM*.gz" 2>/dev/null | wc -l)
if [ "$n_sorted" -lt 2 ]; then
    echo "  NOTE: Run R/01_data_prep.R to sort samples into mgus/mm/normal dirs."
fi

# === External datasets ===
echo ""
echo "====== EXTERNAL DATASETS ======"

download_dataset "GSE33685" "$DATA_DIR/GSE33685" \
    "67 MM + 6 MGUS, Agilent GPL10152" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE33nnn/GSE33685/matrix/GSE33685_series_matrix.txt.gz"

# === Summary ===
echo ""
echo "================================================"
echo " Download Summary"
echo "================================================"
echo ""
for ds in GSE77975 GSE33685; do
    dir="$DATA_DIR/$ds"
    if [ -f "$dir/${ds}_RAW.tar" ]; then
        size=$(du -sh "$dir/${ds}_RAW.tar" | cut -f1)
        n=$(find "$dir/raw" -maxdepth 1 -type f 2>/dev/null | wc -l)
        echo "  $ds: $size tar, $n extracted files"
    else
        echo "  $ds: NOT DOWNLOADED"
    fi
done
echo ""
echo "Data directory structure:"
echo "  data/"
echo "  ├── GSE77975/          (primary — Agilent GPL11358)"
echo "  │   ├── GSE77975_RAW.tar"
echo "  │   ├── GSE77975-GPL*_series_matrix.txt.gz"
echo "  │   └── raw/{mgus,mm,excluded}/"
echo "  ├── GSE33685/          (Agilent GPL10152)"
echo "  │   ├── GSE33685_RAW.tar"
echo "  │   ├── GSE33685_series_matrix.txt.gz"
echo "  │   └── raw/{mgus,mm}/"
echo "  └── output/features/   (merged arm-level features)"
echo ""
echo "Next steps:"
echo "  python scripts/sort_samples.py           # Sort samples by label"
echo "  python python/01_process_all_datasets.py  # Parse, HMM, features"
echo "  python python/04_classification.py        # Classification"
