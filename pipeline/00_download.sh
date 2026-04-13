#!/bin/bash
set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$BASE_DIR/data"

download_dataset() {
    local acc="$1"
    local dest="$2"
    local desc="$3"
    shift 3
    local matrix_urls=("$@")

    echo ""
    echo "--- $acc ($desc) ---"
    mkdir -p "$dest" "$dest/raw"

    if [ ! -f "$dest/${acc}_RAW.tar" ]; then
        echo "  Downloading ${acc}_RAW.tar..."
        curl -L --retry 3 --retry-delay 5 -o "$dest/${acc}_RAW.tar" \
            "https://www.ncbi.nlm.nih.gov/geo/download/?acc=${acc}&format=file"
    else
        echo "  ${acc}_RAW.tar already exists"
    fi

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

    for url in "${matrix_urls[@]}"; do
        local fname
        fname=$(basename "$url")
        if [ ! -f "$dest/$fname" ]; then
            echo "  Downloading $fname..."
            curl -L --retry 3 --retry-delay 5 -o "$dest/$fname" "$url"
        else
            echo "  $fname already exists"
        fi
    done
}

echo "================================================"
echo " MM/MGUS Project — GEO Data Download"
echo "================================================"
echo "Data directory: $DATA_DIR"

# GSE77975: Primary dataset (Agilent 1M, multi-platform)
download_dataset "GSE77975" "$DATA_DIR/GSE77975" \
    "90 MGUS + 33 MM, Agilent multi-platform" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77975/matrix/GSE77975-GPL11358_series_matrix.txt.gz" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77975/matrix/GSE77975-GPL16237_series_matrix.txt.gz" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77975/matrix/GSE77975-GPL10152_series_matrix.txt.gz" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77975/matrix/GSE77975-GPL21461_series_matrix.txt.gz"

for subdir in mgus mm mm_excluded normal; do
    mkdir -p "$DATA_DIR/GSE77975/raw/$subdir"
done

# GSE33685: Japanese cohort (Agilent 60K)
download_dataset "GSE33685" "$DATA_DIR/GSE33685" \
    "67 MM + 6 MGUS, Agilent GPL10152" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE33nnn/GSE33685/matrix/GSE33685_series_matrix.txt.gz"

echo ""
echo "================================================"
echo " Summary"
echo "================================================"
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
