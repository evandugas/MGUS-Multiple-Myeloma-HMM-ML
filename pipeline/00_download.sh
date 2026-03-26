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

# GSE31339: Affymetrix SNP 6.0 (20 MGUS + 20 SMM + 34 MM + 10 normals)
download_dataset "GSE31339" "$DATA_DIR/GSE31339" \
    "20 MGUS + 34 MM, Affymetrix SNP 6.0" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE31nnn/GSE31339/matrix/GSE31339_series_matrix.txt.gz"

# GSE26849: MMRC reference collection (Agilent 244K)
download_dataset "GSE26849" "$DATA_DIR/GSE26849" \
    "254 MM, Agilent 244K" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE26nnn/GSE26849/matrix/GSE26849_series_matrix.txt.gz"

# GSE44745: Malaysian multi-ethnic cohort (Agilent 244K)
download_dataset "GSE44745" "$DATA_DIR/GSE44745" \
    "63 MM, Agilent 244K" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE44nnn/GSE44745/matrix/GSE44745_series_matrix.txt.gz"

# GSE29023: Paired expression+CGH (Agilent 244K)
download_dataset "GSE29023" "$DATA_DIR/GSE29023" \
    "92 MM, Agilent 244K" \
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE29nnn/GSE29023/matrix/GSE29023_series_matrix.txt.gz"

echo ""
echo "================================================"
echo " Summary"
echo "================================================"
for ds in GSE77975 GSE33685 GSE31339 GSE26849 GSE44745 GSE29023; do
    dir="$DATA_DIR/$ds"
    if [ -f "$dir/${ds}_RAW.tar" ]; then
        size=$(du -sh "$dir/${ds}_RAW.tar" | cut -f1)
        n=$(find "$dir/raw" -maxdepth 1 -type f 2>/dev/null | wc -l)
        echo "  $ds: $size tar, $n extracted files"
    else
        echo "  $ds: NOT DOWNLOADED"
    fi
done
