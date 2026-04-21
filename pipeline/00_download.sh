#!/bin/bash
set -e

BASE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$BASE_DIR/data"

echo "================================================"
echo " MM/MGUS Project — GEO Data Download"
echo "================================================"
echo "Data directory: $DATA_DIR"

DS_DIR="$DATA_DIR/GSE77975"
mkdir -p "$DS_DIR" "$DS_DIR/raw"

# Download raw data tar
if [ ! -f "$DS_DIR/GSE77975_RAW.tar" ]; then
    echo "  Downloading GSE77975_RAW.tar..."
    curl -L --retry 3 --retry-delay 5 -o "$DS_DIR/GSE77975_RAW.tar" \
        "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE77975&format=file"
else
    echo "  GSE77975_RAW.tar already exists"
fi

# Extract
n_files=$(find "$DS_DIR/raw" -maxdepth 1 -type f 2>/dev/null | wc -l)
if [ "$n_files" -lt 2 ]; then
    echo "  Extracting..."
    tar -xf "$DS_DIR/GSE77975_RAW.tar" -C "$DS_DIR/raw/"
    n_files=$(find "$DS_DIR/raw" -maxdepth 1 -type f 2>/dev/null | wc -l)
    echo "  Extracted $n_files files"
else
    echo "  Already extracted ($n_files files)"
fi

# Download series matrix files (all 4 platforms)
for gpl in GPL11358 GPL10152 GPL16237 GPL21461; do
    fname="GSE77975-${gpl}_series_matrix.txt.gz"
    if [ ! -f "$DS_DIR/$fname" ]; then
        echo "  Downloading $fname..."
        curl -L --retry 3 --retry-delay 5 -o "$DS_DIR/$fname" \
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE77nnn/GSE77975/matrix/$fname"
    else
        echo "  $fname already exists"
    fi
done

# Create subdirectories for sorting
for subdir in mgus mm excluded; do
    mkdir -p "$DS_DIR/raw/$subdir"
done

echo ""
echo "================================================"
echo " Summary"
echo "================================================"
size=$(du -sh "$DS_DIR/GSE77975_RAW.tar" | cut -f1)
n=$(find "$DS_DIR/raw" -maxdepth 1 -type f 2>/dev/null | wc -l)
echo "  GSE77975: $size tar, $n extracted files"
echo "  Platforms: GPL11358 (CGH+SNP 1M), GPL16237 (180K), GPL10152 (60K), GPL21461 (60K v2)"
